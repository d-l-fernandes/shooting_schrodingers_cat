from __future__ import annotations
from functools import reduce
from typing import NamedTuple, List, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.autograd.functional as autograd
from absl import flags
from matplotlib.figure import Figure

from sde import drifts, diffusions, prior_sdes, variational, priors, sinkhorn
from weak_solver.sdeint import integrate, integrate_parallel_time_steps
from stein import kernel


flags.DEFINE_integer("num_steps", 10, "Number of time steps", lower_bound=1)
flags.DEFINE_integer("num_steps_val", 100, "Number of time steps at validation", lower_bound=1)
flags.DEFINE_integer("num_iter", 50, "Number of IPFP iterations", lower_bound=1)
flags.DEFINE_integer("num_epochs", 10, "Number of epochs.")
flags.DEFINE_integer("batch_repeats", 20, "Optimizer steps per batch", lower_bound=1)
flags.DEFINE_integer("num_samples", 5, "Number of one-step samples", lower_bound=1)

flags.DEFINE_float("final_time", 1., "Final time.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate of the optimizer.")
flags.DEFINE_float("learning_rate_var", 1e-3, "Learning rate of the optimizer.")
flags.DEFINE_float("schedule_scale", 0.1**0.2, "Learning rate scheduler scale.")
flags.DEFINE_float("schedule_iter", 0, "Learning rate scheduler iterations.")
flags.DEFINE_float("grad_clip", 1., "Norm of gradient clip to use.")
flags.DEFINE_enum("solver", "rossler", ["em", "srk", "rossler"], "Solver to use")
flags.DEFINE_enum("solver_val", "rossler", ["em", "srk", "rossler"], "Solver to use in validation")

flags.DEFINE_bool("do_dsb", False, "Whether to use dsb.")

flags.DEFINE_float("initial_sigma", 1., "STD to use in Gaussian (fixed or initial value).")
flags.DEFINE_float("min_sigma", 0.001, "Min STD to use in Gaussian (fixed or initial value).")

FLAGS = flags.FLAGS

Tensor = torch.Tensor

solver_scale = {"em": 1., "srk": 1.5, "rossler": 2.}


class Metrics(NamedTuple):
    wasserstein_prior: Tensor
    wasserstein_data: Tensor
    wasserstein_total: Tensor
    mean_prior: Tensor
    mean_data: Tensor
    std_prior: Tensor
    std_data: Tensor

    def __add__(self, other: Metrics):
        return Metrics(torch.cat((self.wasserstein_prior, other.wasserstein_prior)),
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
            torch.cat((self.x_prior, other.x_prior))
        )


class Model(pl.LightningModule):
    def __init__(self, observed_dims: int, first: bool, forward: bool, results_folder: str, max_diffusion: float):
        super().__init__()

        self.first = first
        self.forward = forward
        self.results_folder = results_folder
        self.max_diffusion = max_diffusion
        self.automatic_optimization = False
        self.observed_dims = observed_dims
        self.metrics = Metrics(torch.tensor([]),
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
        self.time_values = torch.linspace(0, self.final_t, FLAGS.num_steps+1,
                                          device=self.device)
        self.time_values_eval = torch.linspace(0, self.final_t, FLAGS.num_steps_val, device=self.device)

        # SDE
        self.drift_forward: drifts.BaseDrift = \
            drifts.drifts_dict[FLAGS.drift](observed_dims, observed_dims)
        self.drift_backward: drifts.BaseDrift = \
            drifts.drifts_dict[FLAGS.drift](observed_dims, observed_dims)
        self.diffusion = diffusions.Scalar(observed_dims, observed_dims, self.final_t, max_diffusion)

        self.backward_sde = prior_sdes.SDE(self.drift_backward, self.diffusion)
        self.forward_sde = prior_sdes.SDE(self.drift_forward, self.diffusion)

        # Prior
        self.prior_sde: prior_sdes.BasePriorSDE = prior_sdes.prior_sdes_dict[FLAGS.prior_sde](observed_dims)
        self.prior_sde.g = lambda t, x: self.diffusion(x, t)

        # Variational q
        self.q_backwards: variational.BaseVariational \
            = variational.variational_dict[FLAGS.variational](observed_dims, observed_dims)
        self.q_forwards: variational.BaseVariational \
            = variational.variational_dict[FLAGS.variational](observed_dims, observed_dims)
        self.likelihood: priors.BasePrior \
            = priors.priors_dict[FLAGS.prior_dist](observed_dims, observed_dims)
        self.p: priors.BasePrior \
            = priors.priors_dict["gaussian"](observed_dims, observed_dims)

        self.solve_sde = None
        self.optim_sde = None
        self.optim_q = None
        self.data_type = None

        # Sigma exponent
        self.ipfp_iteration = 0
        self.sigma = FLAGS.initial_sigma

        self.get_drift_diffusion(self.first, self.forward)
        self.optim_dict_conv = {"prior": 0, "data": 1}

    def get_drift_diffusion(self, first: bool, forward: bool):
        if self.first:
            self.solve_sde = self.prior_sde
            self.optim_sde = self.backward_sde
            self.optim_q = self.q_backwards
            self.data_type = "prior"
        elif self.forward:
            self.solve_sde = self.forward_sde
            self.optim_sde = self.backward_sde
            self.optim_q = self.q_backwards
            self.data_type = "prior"
        else:
            self.solve_sde = self.backward_sde
            self.optim_sde = self.forward_sde
            self.optim_q = self.q_forwards
            self.data_type = "data"

        self.sigma = max(FLAGS.initial_sigma / 2**self.ipfp_iteration, FLAGS.min_sigma)

    @staticmethod
    def solve(x_0: Tensor, sde, time_values, parallel_time_steps=False, method="em") -> Tensor:
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
        sigma = torch.tensor(self.sigma, device=ys.device)

        q_prob: torch.distributions.Distribution = self.optim_q(ys[:-1], sigma)
        s_is = torch.tile(q_prob.sample().unsqueeze(1), (1, FLAGS.num_samples, 1, 1))
        # s_is = torch.tile(ys[:-1].unsqueeze(1), (1, FLAGS.num_samples, 1, 1))

        time_values = self.time_values.to(ys.device).to(ys.dtype)
        xs = self.solve(s_is, sde, time_values, parallel_time_steps=True, method=FLAGS.solver)

        s_is = s_is.permute(1, 0, 2, 3)
        xs = xs.permute(1, 0, 2, 3)

        # Likelihood
        likelihood_dist = self.likelihood(xs, sigma)
        likelihood = likelihood_dist.log_prob(ys[1:].unsqueeze(0)).mean(0)

        # Variational KL
        prior_dist = self.p(ys[:-1], sigma)
        variational_kl = torch.distributions.kl.kl_divergence(q_prob, prior_dist)

        # Variational KSD
        s_is = s_is.permute(1, 2, 0, 3)
        xs = xs.permute(1, 2, 0, 3)
        transition_density = self.prior_sde.transition_density(
            time_values, s_is, self.forward)
        grad_transition = autograd.jacobian(lambda x: transition_density.log_prob(x).sum(), xs, create_graph=True)

        ksd = kernel.stein_discrepancy(xs, grad_transition)
        delta_ts = time_values[1:] - time_values[:-1]
        diff_values = sde.g(time_values[:-1], s_is)[:, 0, 0].mean(-1)
        max_diff = torch.sqrt(torch.square(self.max_diffusion).sum())

        # ksd_scaled = ksd * self.delta_t**solver_scale[FLAGS.solver]
        # ksd_scaled = torch.einsum(
        #     "a...,a...->a...",
        #     self.final_t**0.5 * max_diff * (delta_ts * diff_values ** 2)**2 / (2 * sigma**2),
        #     ksd)
        ksd_scaled = torch.einsum(
            "a...,a...->a...",
            (delta_ts * diff_values ** 2)**2 / (2 * sigma**2),
            ksd)
        # ksd_scaled = torch.einsum("a...,a...->a...", delta_ts * diff_values ** 2 / (2 * self.sigma**2), ksd)
        # ksd_scaled = torch.einsum("a...,a...->a...", delta_ts**0.5 * diff_values / (self.sigma**2), ksd)  # TOO STRONG
        # ksd_scaled = torch.einsum("a...,a...->a...", delta_ts * self.max_diffusion ** 2 / (self.sigma**2), ksd)
        # ksd_scaled = torch.einsum("a...,a...->a...", self.final_t**0.5 * diff_values / self.sigma, ksd)
        # ksd_scaled = 1 / self.sigma**2 * ksd
        # ksd_scaled = self.max_diffusion / self.sigma**2 * ksd
        # ksd_scaled = torch.einsum("a...,a...->a...", delta_ts**2 / self.sigma**2, ksd)  # TOO WEAK
        # ksd_scaled = torch.einsum("a...,a...->a...", diff_values ** 4 / self.sigma**2, ksd)
        # ksd_scaled = torch.einsum("a...,a...->a...", diff_values ** 4, ksd)
        # ksd_scaled = torch.einsum("a...,a...->a...", diff_values ** 2, ksd)
        # ksd_scaled = torch.einsum("a...,a...->a...", diff_values, ksd)
        # ksd_scaled = torch.einsum("a,a...->a...", delta_ts**0.5 * diff_values, ksd)
        # ksd_scaled = torch.einsum("a,a...->a...", delta_ts * diff_values**2, ksd)
        # ksd_scaled = torch.einsum("a,a...->a...", (delta_ts**0.5 * diff_values)**4, ksd)
        # ksd_scaled = ksd  # TOO WEAK

        # ksd_on = int(not (self.forward and not bool(self.ipfp_iteration)))
        # ksd_scaled = ksd_on * ksd_scaled

        obj = (likelihood - variational_kl - ksd_scaled)
        metrics = {"likelihood": likelihood.mean(), "variational_kl": variational_kl.mean(), "ksd": ksd.mean(),
                   "ksd_scaled": ksd_scaled.mean(),
                   "obj": obj.mean()}
        # obj = (likelihood - ksd_scaled)
        # metrics = {"likelihood": likelihood.mean(),  "ksd": ksd.mean(),
        #            "ksd_scaled": ksd_scaled.mean(),
        #            "obj": obj.mean()}
        return -obj.sum(0).mean(), metrics

    def loss_dsb(self, ys: Tensor, sde, other_sde):
        ys = torch.flip(ys, [0])

        obj = ys[:-1] - ys[1:]
        time_values = self.time_values.to(ys.device).to(ys.dtype)
        drifts_term = \
            sde.f(time_values[:-1], ys[:-1]) - other_sde.f(time_values[:-1], ys[:-1]) + \
            other_sde.f(time_values[1:], ys[1:])

        # TODO replace delta_t with difference of t_values
        obj = (obj + self.delta_t * drifts_term) ** 2
        metrics = {"obj": obj.mean()}
        return obj.mean(), metrics

    def evaluate(self, x_prior: Tensor, x_data: Tensor) -> Output:
        if self.trainer.sanity_checking:
            z_backward = self.solve(x_data, self.prior_sde, self.time_values_eval, method=FLAGS.solver_val)
            z_forward = self.solve(x_prior, self.prior_sde, self.time_values_eval, method=FLAGS.solver_val)
        else:
            z_backward = self.solve(x_data, self.backward_sde, self.time_values_eval, method=FLAGS.solver_val)
            z_forward = self.solve(x_prior, self.forward_sde, self.time_values_eval, method=FLAGS.solver_val)

        output = Output(
            z_backward.permute(1, 0, 2),
            z_forward.permute(1, 0, 2),
            x_data,
            x_prior
        )

        return output

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()[self.optim_dict_conv[self.data_type]]
        x = batch[self.data_type]

        with torch.no_grad():
            delta_ts = torch.rand(FLAGS.num_steps, device=x.device)
            delta_ts = delta_ts / delta_ts.sum() * self.final_t
            self.time_values = torch.cat((torch.tensor([0.], device=x.device), torch.cumsum(delta_ts, 0)))
            xs = self.solve(x, self.solve_sde, self.time_values, method=FLAGS.solver)
            self.time_values = torch.abs(torch.flip(self.final_t - self.time_values, [0]))

        for _ in range(FLAGS.batch_repeats):
            optim.zero_grad(set_to_none=True)
            if FLAGS.do_dsb:
                loss, metrics = self.loss_dsb(xs, self.optim_sde, self.solve_sde)
            else:
                loss, metrics = self.loss(xs, self.optim_sde)
            self.manual_backward(loss)
            # torch.nn.utils.clip_grad_norm_(self.parameters(), FLAGS.grad_clip)
            optim.step()
            # torch.cuda.empty_cache()
        self.log("training", metrics, on_step=True, on_epoch=False, add_dataloader_idx=False)

    def on_train_epoch_end(self, unused=None):
        if (self.current_epoch+1) % FLAGS.num_iter == 0:
            self.forward = not self.forward
            if self.first:
                self.first = False

            if (self.current_epoch + 1) % (FLAGS.num_iter * 2) == 0:
                self.ipfp_iteration += 1
            self.get_drift_diffusion(self.first, self.forward)
            torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        x_prior = batch["prior"]
        x_data = batch["data"]
        output = self.evaluate(x_prior, x_data)
        return output

    def validation_epoch_end(self, outputs: List[Output]):
        final_output: Output = reduce(lambda x, y: x + y, outputs)

        prior_wasserstein = self.wasserstein_loss(final_output.x_prior, final_output.z_values_backward[:, -1])
        data_wasserstein = self.wasserstein_loss(final_output.x_data, final_output.z_values_forward[:, -1])
        total_wasserstein = prior_wasserstein + data_wasserstein

        metrics = Metrics(torch.tensor([prior_wasserstein]),
                          torch.tensor([data_wasserstein]),
                          torch.tensor([total_wasserstein]),
                          final_output.z_values_backward[:, -1].mean(0, keepdim=True).cpu(),
                          final_output.z_values_forward[:, -1].mean(0, keepdim=True).cpu(),
                          final_output.z_values_backward[:, -1].std(0, keepdim=True).cpu(),
                          final_output.z_values_forward[:, -1].std(0, keepdim=True).cpu(),
                          )
        if not self.trainer.sanity_checking:
            log_metrics = {"wasserstein_prior": prior_wasserstein, "wasserstein_data": data_wasserstein,
                           "wasserstein_total": total_wasserstein}
            self.log("eval", log_metrics)
            self.log("wasserstein_total", total_wasserstein)
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
        checkpoint["forward"] = self.forward
        checkpoint["ipfp_iteration"] = self.ipfp_iteration

    def on_load_checkpoint(self, checkpoint):
        self.metrics = checkpoint["metrics"]
        self.first = checkpoint["first"]
        self.forward = checkpoint["forward"]
        self.ipfp_iteration = checkpoint["ipfp_iteration"]

        self.get_drift_diffusion(self.first, self.forward)

    def configure_optimizers(self):
        optim_backward = torch.optim.AdamW([
            {"params": self.diffusion_backward.parameters(), "lr": FLAGS.learning_rate},
            {"params": self.drift_backward.parameters(), "lr": FLAGS.learning_rate},
            {"params": self.q_backwards.parameters(), "lr": FLAGS.learning_rate_var},
        ])
        optim_forward = torch.optim.AdamW([
            {"params": self.diffusion_forward.parameters(), "lr": FLAGS.learning_rate},
            {"params": self.drift_forward.parameters(), "lr": FLAGS.learning_rate},
            {"params": self.q_forwards.parameters(), "lr": FLAGS.learning_rate_var},
        ])
        return optim_backward, optim_forward

    def make_plot_figs(self, output: Output,
                       step: Union[int, str]) -> None:

        fig_list, name_list = self.trainer.datamodule.plot_results(output, self, self.metrics)
        self.save_figs(name_list, fig_list, step)
        plt.clf()
        plt.close("all")
        del fig_list
        del name_list
        del output
        # gc.collect()

    def save_figs(self,
                  fig_names: List[str],
                  figs: List[Figure],
                  step: Union[int, str]) -> None:

        if type(step) == int:
            name = f"step_{step}"
        else:
            name = step

        final = len(fig_names)
        for i in range(final-1, -1, -1):
            figs[i].savefig(f"{self.results_folder}{fig_names[i]}_{name}.png")
            figs[i].clf()
            plt.clf()
            plt.cla()
            plt.close(figs[i])
            del figs[i]

        figs.clear()
