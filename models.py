import gc
from functools import reduce
from typing import NamedTuple, List, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.autograd.functional as functional
from absl import flags
from matplotlib.figure import Figure
import geomloss
import torchsde

from sde import drifts, diffusions, prior_sdes, variational
from stein import kernel


flags.DEFINE_integer("num_steps", 20, "Number of time steps", lower_bound=1)
flags.DEFINE_integer("num_iter", 50, "Number of IPFP iterations", lower_bound=1)
flags.DEFINE_integer("batch_repeats", 1, "Optimizer steps per batch", lower_bound=1)

flags.DEFINE_float("delta_t", 0.01, "Time-step size.")
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
    z_generated: Tensor
    x_data: Tensor
    x_prior: Tensor

    def __add__(self, other):
        return Output(
            torch.cat((self.z_values_backward, other.z_values_backward)),
            torch.cat((self.z_values_forward, other.z_values_forward)),
            torch.cat((self.z_generated, other.z_generated)),
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
        self.metrics = Metrics(torch.tensor([]).detach().cpu(),
                               torch.tensor([]).detach().cpu(),
                               torch.tensor([]).detach().cpu())
        self.wasserstein_loss = geomloss.SamplesLoss()
        self.save_hyperparameters()

        # Prior
        self.prior_sde: prior_sdes.BasePriorSDE = prior_sdes.prior_sdes_dict[FLAGS.prior_sde](observed_dims)

        # SDE
        self.drift_forward: drifts.BaseDrift = \
            drifts.drifts_dict[FLAGS.drift](observed_dims, observed_dims)
        self.diffusion_forward: diffusions.BaseDiffusion = \
            diffusions.diffusions_dict[FLAGS.diffusion](observed_dims, observed_dims)
        self.drift_backward: drifts.BaseDrift = \
            drifts.drifts_dict[FLAGS.drift](observed_dims, observed_dims)
        self.diffusion_backward: diffusions.BaseDiffusion = \
            diffusions.diffusions_dict[FLAGS.diffusion](observed_dims, observed_dims)

        self.sde = None

        if FLAGS.same_diffusion:
            self.diffusion_backward = self.diffusion_forward

        # Variational q
        self.q = variational.variational_dict[FLAGS.variational](observed_dims, observed_dims)

        # Delta_t
        self.time_values = torch.linspace(0, FLAGS.delta_t * FLAGS.num_steps, FLAGS.num_steps+1,
                                          device=self.device)

        self.solve_dict = dict()
        self.optim_dict = dict()

        self.get_drift_diffusion(self.first, self.forward)
        self.optim_dict_conv = {"prior": 0, "data": 1}

    def get_drift_diffusion(self, first: bool, forward: bool):
        if self.first:
            self.solve_dict["drift"] = self.prior_sde.drift
            self.solve_dict["diffusion"] = self.prior_sde.diffusion
            self.optim_dict["drift"] = self.drift_backward
            self.optim_dict["diffusion"] = self.diffusion_backward
            self.data_type = "prior"
        elif self.forward:
            self.solve_dict["drift"] = self.drift_forward
            self.solve_dict["diffusion"] = self.diffusion_forward
            self.optim_dict["drift"] = self.drift_backward
            self.optim_dict["diffusion"] = self.diffusion_backward
            self.data_type = "prior"
        else:
            self.solve_dict["drift"] = self.drift_backward
            self.solve_dict["diffusion"] = self.diffusion_backward
            self.optim_dict["drift"] = self.drift_forward
            self.optim_dict["diffusion"] = self.diffusion_forward
            self.data_type = "data"

        self.sde = prior_sdes.SDE(self.solve_dict["drift"], self.solve_dict["diffusion"])

    def solve(self, x_0: Tensor, drift, diffusion, time_values) -> Tensor:

        self.sde.f = lambda t, x: drift(x, t)
        self.sde.g = lambda t, x: diffusion(x, t)

        xs = torchsde.sdeint(self.sde, x_0, time_values, adaptive=False, dt=FLAGS.delta_t)

        return xs

    def loss(self, ys: Tensor, drift, diffusion, optimize_y):
        ys = torch.flip(ys, [0])

        # grad_log_prob_y = functional.jacobian(
        #     lambda x: q_prob.log_prob(x).sum(), ys, create_graph=True, vectorize=True)
        # obj_y = kernel.stein_discrepancy(ys, grad_log_prob_y)

        if optimize_y:
            q_prob = self.q(ys,
                            FLAGS.delta_t * diffusion(ys, self.time_values[:, None, None].repeat(1, ys.shape[1], 1)))
            obj_y = q_prob.log_prob(ys).mean(1).sum()
            return -obj_y
        else:
            q_prob = self.q(
                ys, FLAGS.delta_t * diffusion(ys, self.time_values[:, None, None].repeat(1, ys.shape[1], 1)).detach())
            # s_is = q_prob.sample()

            xs = torch.empty_like(ys, device=self.device)
            # xs[0] = s_is[0]
            xs[0] = ys[0]

            for i in range(ys.shape[0] - 1):
                # new_xs = self.solve(s_is[i], drift, diffusion, self.time_values[i:i + 2], -1 if self.forward else 1.)
                new_xs = self.solve(ys[i], drift, diffusion, self.time_values[i:i + 2])# , -1 if self.forward else 1.)
                xs[i + 1] = new_xs[-1]

            grad_log_prob_x = functional.jacobian(
                lambda x: q_prob.log_prob(x).sum(), xs, create_graph=True, vectorize=True)

            obj_x = kernel.stein_discrepancy(xs[1:], grad_log_prob_x[1:])
            return obj_x

    def evaluate(self, x_prior: Tensor, x_data: Tensor) -> Output:
        z_backward = self.solve(x_data, self.drift_backward, self.diffusion_backward, self.time_values)
        z_forward = self.solve(z_backward[-1], self.drift_forward, self.diffusion_forward, self.time_values)

        z_generated = self.solve(x_prior, self.drift_forward, self.diffusion_forward, self.time_values)

        output = Output(
            z_backward.permute(1, 0, 2),
            z_forward.permute(1, 0, 2),
            z_generated.permute(1, 0, 2),
            x_data,
            x_prior
        )

        return output

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()[self.optim_dict_conv[self.data_type]]
        optim_q = self.optimizers()[-1]
        x = batch[self.data_type]
        with torch.no_grad():
            xs = self.solve(x, self.solve_dict["drift"], self.solve_dict["diffusion"], self.time_values).detach()

        # for _ in range(FLAGS.batch_repeats):
        #     optim_q.zero_grad(set_to_none=True)
        #     loss = self.loss(xs, self.optim_dict["drift"], self.optim_dict["diffusion"], True)
        #     self.manual_backward(loss)
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), FLAGS.grad_clip)
        #     optim_q.step()
        for _ in range(FLAGS.batch_repeats):
            optim.zero_grad(set_to_none=True)
            loss = self.loss(xs, self.optim_dict["drift"], self.optim_dict["diffusion"], False)
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), FLAGS.grad_clip)
            optim.step()
        # torch.cuda.empty_cache()

    def on_train_epoch_end(self, unused=None):
        if (self.current_epoch+1) % FLAGS.num_iter == 0:
            self.forward = not self.forward
            if self.first:
                self.first = False
        # torch.cuda.empty_cache()
        # gc.collect()

        self.get_drift_diffusion(self.first, self.forward)

    def validation_step(self, batch, batch_idx):
        x_prior = batch["prior"]
        x_data = batch["data"]
        output = self.evaluate(x_prior, x_data)
        return output

    def validation_epoch_end(self, outputs: List[Output]):
        final_output: Output = reduce(lambda x, y: x + y, outputs)

        prior_wasserstein = self.wasserstein_loss(final_output.x_prior, final_output.z_values_forward[:, 0])
        data_wasserstein = self.wasserstein_loss(final_output.x_data, final_output.z_generated[:, -1])
        total_wasserstein = prior_wasserstein + data_wasserstein

        metrics = Metrics(torch.tensor([prior_wasserstein.detach().cpu()]),
                          torch.tensor([data_wasserstein.detach().cpu()]),
                          torch.tensor([total_wasserstein.detach().cpu()]))
        if not self.trainer.sanity_checking:
            self.log("wasserstein_prior", prior_wasserstein, prog_bar=True, sync_dist=True)
            self.log("wasserstein_data", data_wasserstein, prog_bar=True, sync_dist=True)
            self.log("wasserstein_total", total_wasserstein,
                     prog_bar=True, sync_dist=True)
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

    def on_load_checkpoint(self, checkpoint):
        self.metrics = checkpoint["metrics"]
        self.first = checkpoint["first"]
        self.forward = checkpoint["forward"]

        self.get_drift_diffusion(self.first, self.forward)

    def configure_optimizers(self):
        optim_backward = torch.optim.Adam([
            {"params": self.diffusion_backward.parameters()}, {"params": self.drift_backward.parameters()}
        ], lr=FLAGS.learning_rate)
        optim_forward = torch.optim.Adam([
            {"params": self.diffusion_forward.parameters()}, {"params": self.drift_forward.parameters()}
        ], lr=FLAGS.learning_rate)
        optim_q = torch.optim.Adam([
            {"params": self.q.parameters()}
        ], lr=FLAGS.learning_rate)
        return optim_backward, optim_forward, optim_q

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
