from __future__ import annotations
from functools import reduce
from typing import NamedTuple, List, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import functorch
from absl import flags
from matplotlib.figure import Figure

from sde import drifts, diffusions, prior_sdes, priors, sinkhorn
from weak_solver.sdeint import integrate, integrate_parallel_time_steps
from stein import kernel

flags.DEFINE_integer("num_steps", 20, "Number of time steps", lower_bound=1)
flags.DEFINE_integer("num_iter", 25, "Number of IPFP iterations", lower_bound=1)
flags.DEFINE_integer("num_epochs", 10, "Number of epochs.")
flags.DEFINE_integer("batch_repeats", 20, "Optimizer steps per batch", lower_bound=1)
flags.DEFINE_integer("num_samples", 5, "Number of one-step samples", lower_bound=1)

flags.DEFINE_float("final_time", 1.0, "Final time.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate of the optimizer.")
flags.DEFINE_float("schedule_scale", 0.1**0.2, "Learning rate scheduler scale.")
flags.DEFINE_float("schedule_iter", 0, "Learning rate scheduler iterations.")
flags.DEFINE_float("grad_clip", 1.0, "Norm of gradient clip to use.")
flags.DEFINE_float("scale", 1.0, "Regularization scale to use", lower_bound=0.0)
flags.DEFINE_float(
    "scale_increment",
    1.0,
    "Increment to regularization scale per IPFP iteration",
    lower_bound=0.0,
)
flags.DEFINE_float("sigma", 1e-3, "STD to use in Gaussian.")

flags.DEFINE_enum("solver", "em", ["em", "srk", "rossler"], "Solver to use")

flags.DEFINE_bool("do_dsb", False, "Whether to use dsb.")
flags.DEFINE_bool("uniform_delta_t", True, "Whether to use uniform delta t")
flags.DEFINE_bool(
    "no_prior_last_step", True, "Whether to set the prior KSD to 0 at last step."
)
flags.DEFINE_bool(
    "use_brownian_initial", True, "Whether to use Brownian motion as initial SDE."
)
flags.DEFINE_bool(
    "apply_prior_initial",
    False,
    "Whether to apply KSD prior in initial IPFP iteration.",
)

FLAGS = flags.FLAGS

Tensor = torch.Tensor


class Metrics(NamedTuple):
    wasserstein_prior: Tensor
    wasserstein_data: Tensor
    wasserstein_total: Tensor
    mean_prior: Tensor
    mean_data: Tensor
    std_prior: Tensor
    std_data: Tensor

    def __add__(self, other: Metrics):
        return Metrics(
            torch.cat((self.wasserstein_prior, other.wasserstein_prior)),
            torch.cat((self.wasserstein_data, other.wasserstein_data)),
            torch.cat((self.wasserstein_total, other.wasserstein_total)),
            torch.cat((self.mean_prior, other.mean_prior)),
            torch.cat((self.mean_data, other.mean_data)),
            torch.cat((self.std_prior, other.std_prior)),
            torch.cat((self.std_data, other.std_data)),
        )


class Output(NamedTuple):
    z_values_backward: Tensor
    z_values_forward: Tensor
    x_data: Tensor
    x_prior: Tensor

    def __add__(self, other):
        return Output(
            torch.cat((self.z_values_backward, other.z_values_backward)),
            torch.cat((self.z_values_forward, other.z_values_forward)),
            torch.cat((self.x_data, other.x_data)),
            torch.cat((self.x_prior, other.x_prior)),
        )


class Model(pl.LightningModule):
    def __init__(
        self,
        observed_dims: int,
        first: bool,
        forward: bool,
        results_folder: str,
        max_diffusion: float,
    ):
        super().__init__()

        self.first = first
        self.going_forward = forward
        self.results_folder = results_folder
        self.max_diffusion = max_diffusion
        self.automatic_optimization = False
        self.observed_dims = observed_dims
        self.metrics = Metrics(
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )
        self.wasserstein_loss = sinkhorn.SinkhornDistance(eps=0.05, max_iter=100)
        self.save_hyperparameters()

        # Time
        self.final_t = FLAGS.final_time
        self.time_values = torch.linspace(
            0, self.final_t, FLAGS.num_steps + 1, device=self.device
        )
        self.time_values_eval = torch.linspace(
            0, self.final_t, FLAGS.num_steps, device=self.device
        )

        # SDE
        self.drift_forward: drifts.BaseDrift = drifts.drifts_dict[FLAGS.drift](
            observed_dims, observed_dims
        )
        self.drift_backward: drifts.BaseDrift = drifts.drifts_dict[FLAGS.drift](
            observed_dims, observed_dims
        )
        self.diffusion = diffusions.Scalar(
            observed_dims, observed_dims, self.final_t, max_diffusion
        )

        self.backward_sde = prior_sdes.SDE(self.drift_backward, self.diffusion)
        self.forward_sde = prior_sdes.SDE(self.drift_forward, self.diffusion)

        # Prior
        self.prior_sde: prior_sdes.BasePriorSDE = prior_sdes.prior_sdes_dict[
            FLAGS.prior_sde
        ](observed_dims)

        if FLAGS.use_brownian_initial:
            self.initial_prior_sde: prior_sdes.BasePriorSDE = (
                prior_sdes.prior_sdes_dict["brownian"](observed_dims)
            )
        else:
            self.initial_prior_sde: prior_sdes.BasePriorSDE = (
                prior_sdes.prior_sdes_dict[FLAGS.prior_sde](observed_dims)
            )

        self.initial_prior_sde.g = lambda t, x: self.diffusion(x, t)
        self.prior_sde.g = lambda t, x: self.diffusion(x, t)

        self.likelihood_backwards: priors.BasePrior = priors.priors_dict[
            FLAGS.prior_dist
        ](observed_dims, observed_dims)
        self.likelihood_forwards: priors.BasePrior = priors.priors_dict[
            FLAGS.prior_dist
        ](observed_dims, observed_dims)

        self.solve_sde = None
        self.optim_sde = None
        self.optim_likelihood = None
        self.data_type = None

        # Sigma exponent
        self.ipfp_iteration = 0
        self.sigma = FLAGS.sigma
        self.scale = 0.0

        self.get_drift_diffusion(self.first, self.going_forward)
        self.optim_dict_conv = {"prior": 0, "data": 1}

    def get_drift_diffusion(self, first: bool, forward: bool):
        if first:
            self.solve_sde = self.initial_prior_sde
            self.optim_sde = self.backward_sde
            self.optim_likelihood = self.likelihood_backwards
            self.data_type = "prior"
        elif forward:
            self.solve_sde = self.forward_sde
            self.optim_sde = self.backward_sde
            self.optim_likelihood = self.likelihood_backwards
            self.data_type = "prior"
        else:
            self.solve_sde = self.backward_sde
            self.optim_sde = self.forward_sde
            self.optim_likelihood = self.likelihood_forwards
            self.data_type = "data"

        if FLAGS.apply_prior_initial:
            self.scale = FLAGS.scale
        else:
            self.scale = min(FLAGS.scale_increment * self.ipfp_iteration, FLAGS.scale)

    @staticmethod
    def solve(
        x_0: Tensor, sde, time_values, parallel_time_steps=False, method="em"
    ) -> Tensor:
        if FLAGS.do_dsb:
            xs = integrate(sde, x_0, time_values, method="em")
        else:
            if parallel_time_steps:
                xs = integrate_parallel_time_steps(sde, x_0, time_values, method=method)
            else:
                xs = integrate(sde, x_0, time_values, method=method)
        return xs

    def loss(self, ys: Tensor, sde):
        ys = torch.flip(ys, [0])

        s_is = torch.tile(ys[:-1].unsqueeze(1), (1, FLAGS.num_samples, 1, 1))

        time_values = self.time_values.to(ys.device).to(ys.dtype)
        xs = self.solve(s_is, sde, time_values, parallel_time_steps=True, method=FLAGS.solver)

        s_is = s_is.permute(0, 2, 1, 3)
        xs = xs.permute(1, 0, 2, 3)

        # Likelihood
        p_ys = self.optim_likelihood(ys[1:], self.sigma)
        p_ys_prior = self.prior_sde.transition_density(time_values, s_is, self.going_forward)

        # Variational KSD
        grad_p_ys = functorch.grad(lambda x: p_ys.log_prob(x).sum())(xs)

        xs = xs.permute(1, 2, 0, 3)
        grad_p_ys_prior = functorch.grad(lambda x: p_ys_prior.log_prob(x).sum())(xs)

        grad_p_ys = grad_p_ys.permute(1, 2, 0, 3)

        ksds = kernel.stein_discrepancy(
            xs, tuple(grad_p_ys, grad_p_ys_prior)
        )

        ksd = ksds[0]
        ksd_prior = ksds[1]
        ksd = ksd.mean(-1)
        ksd_prior = ksd_prior.mean(-1)

        delta_ts = time_values[1:] - time_values[:-1]
        diffusions = sde.g(time_values[:-1], s_is)

        ksd_prior = torch.einsum(
            "a,a...->a...",
            (delta_ts**0.5 * diffusions) ** 4 / self.sigma**4,
            ksd_prior,
        )

        if FLAGS.no_prior_last_step:
            ksd_prior[-1] = ksd_prior[-1] * 0.0

        obj = ksd.sum() + self.scale * ksd_prior.sum()
        metrics = {
            "ksd": ksd.sum() * self.sigma**4,
            "ksd_prior": ksd_prior.sum() * self.sigma**4,
            "obj": obj * self.sigma**4,
        }
        return obj, metrics

    def loss_dsb(self, ys: Tensor, sde, other_sde):
        ys = torch.flip(ys, [0])

        obj = ys[:-1] - ys[1:]
        time_values = self.time_values.to(ys.device).to(ys.dtype)
        drifts_term = (
            sde.f(time_values[:-1], ys[:-1])
            - other_sde.f(time_values[:-1], ys[:-1])
            + other_sde.f(time_values[1:], ys[1:])
        )

        # TODO replace delta_t with difference of t_values
        obj = (obj + self.delta_t * drifts_term) ** 2
        metrics = {"obj": obj.mean()}
        return obj.mean(), metrics

    def training_step(self, batch, _):
        optim = self.optimizers()[self.optim_dict_conv[self.data_type]]
        x = batch[self.data_type]

        with torch.no_grad():
            if FLAGS.uniform_delta_t:
                xs = self.solve(
                    x, self.solve_sde, self.time_values, method=FLAGS.solver
                )
            else:
                delta_ts = torch.rand(FLAGS.num_steps, device=x.device)
                delta_ts = delta_ts / delta_ts.sum() * self.final_t
                self.time_values = torch.cat(
                    (torch.tensor([0.0], device=x.device), torch.cumsum(delta_ts, 0))
                ).detach()
                xs = self.solve(
                    x, self.solve_sde, self.time_values, method=FLAGS.solver
                )
                self.time_values = torch.abs(
                    torch.flip(self.final_t - self.time_values, [0])
                )

        for _ in range(FLAGS.batch_repeats):
            optim.zero_grad()
            if FLAGS.do_dsb:
                loss, metrics = self.loss_dsb(xs, self.optim_sde, self.solve_sde)
            else:
                loss, metrics = self.loss(xs, self.optim_sde)
            self.manual_backward(loss)
            # torch.nn.utils.clip_grad_norm_(self.parameters(), FLAGS.grad_clip)
            optim.step()
        self.log(
            "training", metrics, on_step=True, on_epoch=False, add_dataloader_idx=False
        )

    def evaluate(self, x_prior: Tensor, x_data: Tensor) -> Output:
        if self.trainer.sanity_checking:
            z_backward = self.solve(
                x_data,
                self.initial_prior_sde,
                self.time_values_eval,
                method=FLAGS.solver,
            )
            z_forward = self.solve(
                x_prior,
                self.initial_prior_sde,
                self.time_values_eval,
                method=FLAGS.solver,
            )
        else:
            z_backward = self.solve(
                x_data,
                self.backward_sde,
                self.time_values_eval,
                method=FLAGS.solver,
            )
            z_forward = self.solve(
                x_prior,
                self.forward_sde,
                self.time_values_eval,
                method=FLAGS.solver,
            )

        output = Output(
            z_backward.permute(1, 0, 2), z_forward.permute(1, 0, 2), x_data, x_prior
        )

        return output

    def on_train_epoch_end(self, _=None):
        if (self.current_epoch + 1) % FLAGS.num_iter == 0:
            self.going_forward = not self.going_forward
            if self.first:
                self.first = False

            if (self.current_epoch + 1) % (FLAGS.num_iter * 2) == 0:
                if self.ipfp_iteration < FLAGS.schedule_iter:
                    lr_scheduler_backward, lr_scheduler_forward = self.lr_schedulers()
                    lr_scheduler_backward.step()
                    lr_scheduler_forward.step()
                self.ipfp_iteration += 1
            self.get_drift_diffusion(self.first, self.going_forward)
            torch.cuda.empty_cache()

    def validation_step(self, batch, _):
        x_prior = batch["prior"]
        x_data = batch["data"]
        output = self.evaluate(x_prior, x_data)
        return output

    def validation_epoch_end(self, outputs: List[Output]):
        final_output: Output = reduce(lambda x, y: x + y, outputs)

        prior_wasserstein = self.wasserstein_loss(
            final_output.x_prior, final_output.z_values_backward[:, -1]
        )
        data_wasserstein = self.wasserstein_loss(
            final_output.x_data, final_output.z_values_forward[:, -1]
        )
        total_wasserstein = prior_wasserstein + data_wasserstein

        metrics = Metrics(
            torch.tensor([prior_wasserstein]),
            torch.tensor([data_wasserstein]),
            torch.tensor([total_wasserstein]),
            final_output.z_values_backward[:, -1].mean(0, keepdim=True).cpu(),
            final_output.z_values_forward[:, -1].mean(0, keepdim=True).cpu(),
            final_output.z_values_backward[:, -1].std(0, keepdim=True).cpu(),
            final_output.z_values_forward[:, -1].std(0, keepdim=True).cpu(),
        )
        if not self.trainer.sanity_checking:
            log_metrics = {
                "wasserstein_prior": prior_wasserstein,
                "wasserstein_data": data_wasserstein,
                "wasserstein_total": total_wasserstein,
            }
            self.log("eval", log_metrics)
            self.log("data", data_wasserstein, prog_bar=True, logger=False)
            self.log("prior", prior_wasserstein, prog_bar=True, logger=False)
            self.log("total", total_wasserstein, prog_bar=True, logger=False)
        self.metrics += metrics

        if self.trainer.sanity_checking:
            self.make_plot_figs(final_output, "sanity")
        elif self.current_epoch == 0:
            self.make_plot_figs(final_output, "predict")
        else:
            self.make_plot_figs(final_output, self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["metrics"] = self.metrics
        checkpoint["first"] = self.first
        checkpoint["forward"] = self.going_forward
        checkpoint["ipfp_iteration"] = self.ipfp_iteration

    def on_load_checkpoint(self, checkpoint):
        self.metrics = checkpoint["metrics"]
        self.first = checkpoint["first"]
        self.going_forward = checkpoint["forward"]
        self.ipfp_iteration = checkpoint["ipfp_iteration"]

        self.get_drift_diffusion(self.first, self.going_forward)

    def configure_optimizers(self):
        optim_backward = torch.optim.Adam(
            [
                {"params": self.drift_backward.parameters(), "lr": FLAGS.learning_rate},
                {
                    "params": self.likelihood_backwards.parameters(),
                    "lr": FLAGS.learning_rate,
                },
            ]
        )
        optim_forward = torch.optim.Adam(
            [
                {"params": self.drift_forward.parameters(), "lr": FLAGS.learning_rate},
                {
                    "params": self.likelihood_forwards.parameters(),
                    "lr": FLAGS.learning_rate,
                },
            ]
        )
        scheduler_backward = torch.optim.lr_scheduler.ExponentialLR(
            optim_backward, FLAGS.schedule_scale
        )
        scheduler_forward = torch.optim.lr_scheduler.ExponentialLR(
            optim_forward, FLAGS.schedule_scale
        )
        return [optim_backward, optim_forward], [scheduler_backward, scheduler_forward]

    def make_plot_figs(self, output: Output, step: Union[int, str]) -> None:

        fig_list, name_list = self.trainer.datamodule.plot_results(
            output, self, self.metrics
        )
        self.save_figs(name_list, fig_list, step)
        plt.clf()
        plt.close("all")
        del fig_list
        del name_list
        del output
        # gc.collect()

    def save_figs(
        self, fig_names: List[str], figs: List[Figure], step: Union[int, str]
    ) -> None:

        if type(step) == int:
            name = f"step_{step}"
        else:
            name = step

        final = len(fig_names)
        for i in range(final - 1, -1, -1):
            figs[i].savefig(f"{self.results_folder}{fig_names[i]}_{name}.png")
            figs[i].clf()
            plt.clf()
            plt.cla()
            plt.close(figs[i])
            del figs[i]

        figs.clear()
