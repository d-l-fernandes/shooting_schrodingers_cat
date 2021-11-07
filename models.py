import gc
from functools import reduce
from typing import NamedTuple, List, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.autograd.functional as autograd
from absl import flags
from matplotlib.figure import Figure
import geomloss
import torchsde

from sde import drifts, diffusions, prior_sdes, variational, priors
from weak_solver.sdeint import integrate
from stein import kernel


flags.DEFINE_integer("num_steps", 10, "Number of time steps", lower_bound=1)
flags.DEFINE_integer("num_iter", 50, "Number of IPFP iterations", lower_bound=1)
flags.DEFINE_integer("batch_repeats", 1, "Optimizer steps per batch", lower_bound=1)
flags.DEFINE_integer("num_samples", 10, "Number of one-step_samples", lower_bound=1)

flags.DEFINE_float("delta_t", 0.05, "Time-step size.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of the optimizer.")
flags.DEFINE_float("grad_clip", 1., "Norm of gradient clip to use.")

flags.DEFINE_bool("same_diffusion", False, "Whether to use same diffusion.")

FLAGS = flags.FLAGS

Tensor = torch.Tensor


class Metrics(NamedTuple):
    wasserstein_prior: Tensor
    wasserstein_data: Tensor
    wasserstein_total: Tensor

    def __add__(self, other):
        return Metrics(torch.cat((self.wasserstein_prior, other.wasserstein_prior)),
                       torch.cat((self.wasserstein_data, other.wasserstein_data)),
                       torch.cat((self.wasserstein_total, other.wasserstein_total)),
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
    def __init__(self, observed_dims: int, first: bool, forward: bool, results_folder: str):
        super().__init__()

        self.first = first
        self.forward = forward
        self.results_folder = results_folder
        self.automatic_optimization = False
        self.metrics = Metrics(torch.tensor([]),
                               torch.tensor([]),
                               torch.tensor([]))
        self.wasserstein_loss = geomloss.SamplesLoss()
        self.save_hyperparameters()

        # Time
        self.final_t = FLAGS.delta_t * FLAGS.num_steps
        self.time_values = torch.linspace(0, self.final_t, FLAGS.num_steps+1,
                                          device=self.device)
        self.delta_t = torch.tensor(FLAGS.delta_t, device=self.device)

        # SDE
        self.drift_forward: drifts.BaseDrift = \
            drifts.drifts_dict[FLAGS.drift](observed_dims, observed_dims)
        self.diffusion_forward: diffusions.BaseDiffusion = \
            diffusions.diffusions_dict[FLAGS.diffusion](observed_dims, observed_dims, self.final_t)
        self.drift_backward: drifts.BaseDrift = \
            drifts.drifts_dict[FLAGS.drift](observed_dims, observed_dims)
        self.diffusion_backward: diffusions.BaseDiffusion = \
            diffusions.diffusions_dict[FLAGS.diffusion](observed_dims, observed_dims, self.final_t)

        if FLAGS.same_diffusion:
            self.diffusion_backward = self.diffusion_forward

        self.forward_sde = prior_sdes.SDE(self.drift_forward, self.diffusion_forward)
        self.backward_sde = prior_sdes.SDE(self.drift_backward, self.diffusion_backward)

        # Prior
        self.prior_sde: prior_sdes.BasePriorSDE = prior_sdes.prior_sdes_dict[FLAGS.prior_sde](observed_dims)
        self.prior_sde.g = lambda t, x: self.diffusion_forward(x, t)

        # Variational q
        self.q_forward: variational.BaseVariational \
            = variational.variational_dict[FLAGS.variational](observed_dims, observed_dims, FLAGS.sigma)
        self.q_backward: variational.BaseVariational \
            = variational.variational_dict[FLAGS.variational](observed_dims, observed_dims, FLAGS.sigma)
        self.likelihood: priors.BasePrior \
            = priors.priors_dict[FLAGS.prior_dist](observed_dims, observed_dims)
        self.p: priors.BasePrior \
            = priors.priors_dict["gaussian"](observed_dims, observed_dims)

        self.solve_sde = None
        self.optim_sde = None
        self.data_type = None
        self.optim_q = None

        self.get_drift_diffusion(self.first, self.forward)
        self.optim_dict_conv = {"prior": 0, "data": 1}

        # Sigma exponent
        self.ipfp_iteration = 0

    def get_drift_diffusion(self, first: bool, forward: bool):
        if self.first:
            self.solve_sde = self.prior_sde
            self.optim_sde = self.backward_sde
            self.optim_q = self.q_backward
            self.data_type = "prior"
        elif self.forward:
            self.solve_sde = self.forward_sde
            self.optim_sde = self.backward_sde
            self.optim_q = self.q_backward
            self.data_type = "prior"
        else:
            self.solve_sde = self.backward_sde
            self.optim_sde = self.forward_sde
            self.optim_q = self.q_forward
            self.data_type = "data"

    @staticmethod
    def solve(x_0: Tensor, sde, time_values) -> Tensor:
        # xs = torchsde.sdeint(sde, x_0, time_values, method="srk", adaptive=False, dt=FLAGS.delta_t)
        xs = integrate(sde, x_0, time_values)
        return xs

    def loss(self, ys: Tensor, sde):
        ys = torch.flip(ys, [0])

        q_prob: torch.distributions.Distribution = self.optim_q(ys[:-1])
        # s_is = q_prob.sample((FLAGS.num_samples,)).detach()
        s_is = q_prob.sample((FLAGS.num_samples,))

        xs = torch.empty_like(s_is, device=self.device)

        for i in range(ys.shape[0]-1):
            xs[:, i] = self.solve(s_is[:, i], sde, self.time_values[i:i+2])[-1]

        # Likelihood
        # index = torch.randint(FLAGS.num_samples, (1,))[0]
        # likelihood_dist = self.likelihood(xs[index])
        # likelihood = likelihood_dist.log_prob(ys[1:])
        likelihood_dist = self.likelihood(xs)
        likelihood = likelihood_dist.log_prob(ys[1:]).mean(0)

        # Variational KL
        prior_dist = self.p(ys[:-1])
        variational_kl = torch.distributions.kl.kl_divergence(q_prob, prior_dist)

        # Variational KSD
        s_is = s_is.permute(1, 2, 0, 3)
        xs = xs.permute(1, 2, 0, 3)
        transition_density = self.prior_sde.transition_density(self.time_values.to(ys.device), s_is, self.forward)
        grad_transition = autograd.jacobian(lambda x: transition_density.log_prob(x).sum(), xs, create_graph=True)
        ksd = kernel.stein_discrepancy(xs, grad_transition, FLAGS.sigma, self.delta_t, self.ipfp_iteration)

        obj = (likelihood - variational_kl - ksd)
        metrics = {"likelihood": likelihood.mean(), "variational_kl": variational_kl.mean(), "ksd": ksd.mean(),
                   "obj": obj.mean()}
        # self.log("likelihood", likelihood.mean())
        # self.log("variational_kl", variational_kl.mean())
        # self.log("ksd", ksd.mean())
        # self.log("obj", obj.mean())
        return -obj.mean(), metrics

    def evaluate(self, x_prior: Tensor, x_data: Tensor) -> Output:
        if self.trainer.sanity_checking:
            z_backward = self.solve(x_data, self.prior_sde, self.time_values)
            z_forward = self.solve(x_prior, self.prior_sde, self.time_values)
        else:
            z_backward = self.solve(x_data, self.backward_sde, self.time_values)
            z_forward = self.solve(x_prior, self.forward_sde, self.time_values)

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
            xs = self.solve(x, self.solve_sde, self.time_values)

        # for _ in range(FLAGS.batch_repeats):
        # optim.zero_grad(set_to_none=True)
        optim.zero_grad()
        loss, metrics = self.loss(xs, self.optim_sde)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), FLAGS.grad_clip)
        optim.step()
        # torch.cuda.empty_cache()
        self.log("training", metrics, on_step=False, on_epoch=True, add_dataloader_idx=False)

    def on_train_epoch_end(self, unused=None):
        if (self.current_epoch+1) % FLAGS.num_iter == 0:
            self.forward = not self.forward
            if self.first:
                self.first = False

            self.get_drift_diffusion(self.first, self.forward)
            torch.cuda.empty_cache()
        if (self.current_epoch+1) % (FLAGS.num_iter*2) == 0:
            self.ipfp_iteration += 1


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
                          torch.tensor([total_wasserstein]))
        if not self.trainer.sanity_checking:
            log_metrics = {"wasserstein_prior": prior_wasserstein, "wasserstein_data": data_wasserstein,
                       "wasserstein_total": total_wasserstein}
            self.log("eval", log_metrics)
            self.log("wasserstein_total", total_wasserstein)
            # self.log("wasserstein_prior", prior_wasserstein, prog_bar=True)
            # self.log("wasserstein_data", data_wasserstein, prog_bar=True)
            # self.log("wasserstein_total", total_wasserstein, prog_bar=True)
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
            {"params": self.diffusion_backward.parameters()}, {"params": self.drift_backward.parameters()},
            {"params": self.q_backward.parameters()},
        ], lr=FLAGS.learning_rate)
        optim_forward = torch.optim.AdamW([
            {"params": self.diffusion_forward.parameters()}, {"params": self.drift_forward.parameters()},
            {"params": self.q_forward.parameters()},
        ], lr=FLAGS.learning_rate)
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
