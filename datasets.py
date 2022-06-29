import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions
import seaborn as sns
import pandas as pd
from absl import flags
from matplotlib import figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from typing import Optional, List, Tuple, Any
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn import datasets
from torchvision import datasets as vision_datasets

from models import Output, Model, Metrics
from sde import prior_sdes

Tensor = torch.Tensor

datasets_list = [
                    # Priors
                    "gaussian",
                    # Experiments
                    "blobs_2d",
                    "blobs_3d",
                    "double_well_left",
                    "double_well_bimodal_left",
                    "double_well_right",
                    "spiral_one",
                    "spiral_two",
                    "s_curve",
                    "swiss_roll",
                    "moon",
                    "circle",
                    "checker",
                    # Bounds experiment
                    "gaussian_bound_left",
                    "gaussian_bound_right",
                    # MNIST,
                    "mnist"
]

flags.DEFINE_integer("batch_size", 1500, "Batch Size.")
# flags.DEFINE_integer("dims", 2, "Number of dims.")
flags.DEFINE_enum("prior", "gaussian", datasets_list, "Prior to use.")
flags.DEFINE_enum("dataset", "blobs_2d", datasets_list, "Dataset to use.")
flags.DEFINE_bool("normalize", True, "Whether to normalize data")
flags.DEFINE_float("prior_scale", 1., "Scale parameter of prior dataset.")

FLAGS = flags.FLAGS


def normalize(data):
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, dim=0, keepdim=True)
    return (data - mean) / std


class BaseDataGenerator(LightningDataModule):
    def __init__(self, prior_dataset=None):
        super().__init__()
        self.batch_size: int = FLAGS.batch_size
        self.prior_dataset = prior_dataset

        # Data properties
        self.n_train: int = 0
        self.n_test: int = 0
        self.observed_dims: int = 0

        # Plotting properties
        self.draw_y_axis: bool = False
        self.x_lims = [[-3, 3], [-3, 3]]

        # Train arrays
        self.xs_train: Tensor = torch.ones((self.n_train, self.observed_dims))
        # Test arrays
        self.xs_test: Tensor = torch.ones((self.n_test, self.observed_dims))

        # torch.manual_seed(42)
        # np.random.seed(42)

        # To calculate max diffusion
        self.dataset_min = 0.
        self.dataset_max = 0.
        self.max_diffusion = 0.

    def calculate_max_diffusion(self):
        if self.prior_dataset is None:
            raise RuntimeError("Needs prior to be able to calculate max diffusion")

        self.dataset_max = torch.max((self.xs_train**2).sum(-1)**0.5)
        self.dataset_min = torch.min((self.xs_train**2).sum(-1)**0.5)
        self.prior_dataset.dataset_max = torch.max((self.prior_dataset.xs_train**2).sum(-1)**0.5)
        self.prior_dataset.dataset_min = torch.min((self.prior_dataset.xs_train**2).sum(-1)**0.5)

        self.max_diffusion = float(max(torch.abs(self.dataset_max - self.prior_dataset.dataset_max).numpy(),
                                       torch.abs(self.dataset_max - self.prior_dataset.dataset_min).numpy()))
        # self.max_diffusion = \
        #     float((torch.std(self.xs_train, unbiased=True) +
        #            (((self.xs_train.mean(0) - self.prior_dataset.xs_train.mean(0))**2).sum()**0.5).numpy()))
        # self.max_diffusion = torch.max(
        #     torch.std(self.xs_train, dim=0, unbiased=True),
        #     (((self.xs_train.mean(0) - self.prior_dataset.xs_train.mean(0))**2)**0.5)
        # ).numpy()
        # self.max_diffusion = torch.std(self.xs_train, unbiased=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Train arrays
        self.xs_train: Tensor = torch.ones((self.n_train, self.observed_dims))
        # Test arrays
        self.xs_test: Tensor = torch.ones((self.n_test, self.observed_dims))

        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def train_dataloader(self):
        if self.prior_dataset is not None:
            if FLAGS.normalize:
                self.prior_dataset.xs_train = normalize(self.prior_dataset.xs_train)
                self.xs_train = normalize(self.xs_train)
            loaders = {
                "prior": DataLoader(
                    FLAGS.prior_scale * self.prior_dataset.xs_train, self.batch_size, shuffle=True, num_workers=0),
                "data": DataLoader(self.xs_train, self.batch_size, shuffle=True, num_workers=0)}
            return CombinedLoader(loaders, "max_size_cycle")
        else:
            if FLAGS.normalize:
                self.xs_train = normalize(self.xs_train)
            return DataLoader(self.xs_train, self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        if self.prior_dataset is not None:
            if FLAGS.normalize:
                self.prior_dataset.xs_test = normalize(self.prior_dataset.xs_test)
                self.xs_test = normalize(self.xs_test)
            loaders = {
                "prior": DataLoader(FLAGS.prior_scale * self.prior_dataset.xs_test, self.batch_size,
                                    num_workers=0, pin_memory=True),
                "data": DataLoader(self.xs_test, self.batch_size, num_workers=0, pin_memory=True)}
            return CombinedLoader(loaders, "max_size_cycle")
        else:
            if FLAGS.normalize:
                self.xs_test = normalize(self.xs_test)
            return DataLoader(self.xs_test, self.batch_size, num_workers=0, pin_memory=True)

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        raise NotImplementedError

    @staticmethod
    def plot_objective(gs_obj: Any, gs_ll: Any, fig_obj: Figure, fig_ll: Figure, metrics: Metrics):
        wasserstein_prior = metrics.wasserstein_prior.cpu().detach().numpy()
        wasserstein_data = metrics.wasserstein_data.cpu().detach().numpy()
        wasserstein_total = metrics.wasserstein_total.cpu().detach().numpy()
        ll_forwards = metrics.prior_ll_forwards.cpu().detach().numpy()
        ll_backwards = metrics.prior_ll_backwards.cpu().detach().numpy()

        # Objective plotting
        epochs_array = np.arange(0, wasserstein_prior.shape[0])
        ax_objective: Axes = fig_obj.add_subplot(gs_obj[0, :])
        ax_ll: Axes = fig_ll.add_subplot(gs_ll[0, :])

        indices = np.arange(0, wasserstein_prior.shape[0])
        ax_objective.plot(epochs_array[indices], wasserstein_prior[indices], c="r", label="prior")
        ax_objective.plot(epochs_array[indices], wasserstein_data[indices], c="b", label="data")
        ax_objective.plot(epochs_array[indices], wasserstein_total[indices], c="k", label="total")
        ax_objective.set_xlabel("Epoch")
        # ax_objective.set_yscale("symlog")
        ax_objective.legend(loc=2)
        ax_objective.grid(True)
        ax_ll.plot(epochs_array[indices], ll_forwards[indices], c="r", label="ll_forwards")
        ax_ll.plot(epochs_array[indices], ll_backwards[indices], c="b", label="ll_backwards")
        ax_ll.set_xlabel("Epoch")
        # ax_ll.set_yscale("symlog")
        ax_ll.legend(loc=2)
        ax_ll.grid(True)

        # if len(indices) > 2:
        #     ax_objective.set_ylim(top=1.1 * (np.sum(wasserstein_total) / len(indices)))
        #     ax_ll_minimum = min(np.sum(ll_forwards) / len(indices),
        #                         np.sum(ll_backwards) / len(indices))
        #     if ax_ll_minimum > 0:
        #         ax_ll_minimum *= 0.99
        #     else:
        #         ax_ll_minimum *= 1.01
        #     ax_ll.set_ylim(bottom=ax_ll_minimum)

        if len(indices) != 0:
            total_min = np.argmin(wasserstein_total)
            ax_objective.vlines(total_min, wasserstein_total[total_min], 1.05 * wasserstein_total[total_min], color="k")
            ax_objective.annotate(f"{wasserstein_total[total_min]:.3f}",
                                  xy=(total_min, 1.05 * wasserstein_total[total_min]),
                                  xytext=(-3, 3), textcoords="offset points", horizontalalignment="right",
                                  verticalalignment="bottom")

    def plot_2d_to_2d(self, output: Output, model: Model, metrics: Metrics) -> \
            Tuple[List[Figure], List[str]]:
        z_values_forward = output.z_values_forward.cpu().detach().numpy()
        z_values_backward = output.z_values_backward.cpu().detach().numpy()
        x_data = output.x_data.cpu().detach().numpy()
        x_prior = output.x_prior.cpu().detach().numpy()

        if z_values_backward.shape[-1] == 2:
            # Values
            z_1 = z_values_forward[:, -1]
            z_0 = z_values_backward[:, -1]

            fig_obj: Figure = figure.Figure(figsize=(15, 15))
            gs = fig_obj.add_gridspec(1, 1, height_ratios=(1,), width_ratios=(1,),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.1, hspace=0.1)

            fig_ll: Figure = figure.Figure(figsize=(15, 15))
            gs_ll = fig_ll.add_gridspec(1, 1, height_ratios=(1,), width_ratios=(1,),
                                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                                        wspace=0.1, hspace=0.1)

            self.plot_objective(gs, gs_ll, fig_obj, fig_ll, metrics)
            # Plotting
            fig_z_backwards: Figure = figure.Figure(figsize=(15, 15))
            ax_z_backwards: Axes = fig_z_backwards.add_subplot(1, 1, 1)
            ax_z_backwards.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_backwards.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            fig_z_forwards: Figure = figure.Figure(figsize=(15, 15))
            ax_z_forwards: Axes = fig_z_forwards.add_subplot(1, 1, 1)
            ax_z_forwards.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_forwards.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_z.axis("off")

            fig_z_0: Figure = figure.Figure(figsize=(15, 15))
            ax_z_0: Axes = fig_z_0.add_subplot(1, 1, 1)
            ax_z_0.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_0.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_x.axis("off")

            fig_z_1: Figure = figure.Figure(figsize=(15, 15))
            ax_z_1: Axes = fig_z_1.add_subplot(1, 1, 1)
            ax_z_1.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_1.set_xlim(self.x_lims[0][0], self.x_lims[0][1])

            t_values = model.time_values_eval.cpu().detach().numpy()

            if type(model.prior_sde) in [prior_sdes.Hill, prior_sdes.Periodic]:
                xx, yy = np.meshgrid(np.linspace(self.x_lims[0][0], self.x_lims[0][1], 100),
                                     np.linspace(self.x_lims[1][0], self.x_lims[1][1], 100))
                zz = model.prior_sde.u(
                    torch.tensor(xx).to(output.z_values_forward.device),
                    torch.tensor(yy).to(output.z_values_forward.device),
                ).detach().cpu().numpy()
                ax_z_1.pcolormesh(xx, yy, zz, alpha=0.6, cmap="binary", shading="auto", rasterized=True)
                ax_z_0.pcolormesh(xx, yy, zz, alpha=0.6, cmap="binary", shading="auto", rasterized=True)
                ax_z_backwards.pcolormesh(xx, yy, zz, alpha=0.6, cmap="binary", shading="auto", rasterized=True)
                ax_z_forwards.pcolormesh(xx, yy, zz, alpha=0.6, cmap="binary", shading="auto", rasterized=True)

            norm = plt.Normalize(t_values.min(), t_values.max())

            ax_z_1.scatter(x_data[:, 0], x_data[:, 1], c="k", marker='o', alpha=0.2)
            ax_z_1.scatter(z_1[:, 0], z_1[:, 1], c="darkorange", marker='o', alpha=0.2)
            ax_z_0.scatter(x_prior[:, 0], x_prior[:, 1], c="k", marker='o', alpha=0.2)
            ax_z_0.scatter(z_0[:, 0], z_0[:, 1], c="darkorange", marker='o', alpha=0.2)

            n_paths = 500
            indices = np.random.choice(z_values_backward.shape[0], n_paths, replace=False)
            for i in range(0, n_paths):
                index = indices[i]
                points = np.array([z_values_forward[index, :, 0], z_values_forward[index, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm, zorder=2)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.2)
                ax_z_forwards.add_collection(lc)
                del lc
            for i in range(0, n_paths):
                index = indices[i]
                points = np.array([z_values_backward[index, :, 0], z_values_backward[index, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm, zorder=2)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.2)
                ax_z_backwards.add_collection(lc)
                del lc

            ax_z_forwards.scatter(x_data[:, 0], x_data[:, 1],  c="k", marker='o',  alpha=0.2, zorder=1)
            ax_z_backwards.scatter(x_prior[:, 0], x_prior[:, 1], c="k", marker='o', alpha=0.2, zorder=1)
            ax_z_backwards.scatter(z_values_backward[:, -1, 0], z_values_backward[:, -1, 1],
                                   c="darkorange",
                                   alpha=0.2, marker="o", zorder=3)
            ax_z_forwards.scatter(z_values_forward[:, -1, 0], z_values_forward[:, -1, 1],
                                  c="darkorange",
                                  alpha=0.2, marker="o", zorder=3)

            return [fig_obj, fig_ll, fig_z_forwards, fig_z_backwards, fig_z_0, fig_z_1], \
                   ["objective", "ll", "forwards", "backwards", "z_0", "z_1"]

        else:
            raise ValueError("Dims must be 2")


class Gaussian(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        # self.observed_dims: int = FLAGS.dims
        self.observed_dims: int = 2

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        self.xs_train = distributions.MultivariateNormal(
            loc=torch.tensor([0.] * self.observed_dims),
            scale_tril=torch.eye(self.observed_dims)).sample((self.n_train,))
        # Test
        self.xs_test = distributions.MultivariateNormal(
            loc=torch.tensor([0.] * self.observed_dims),
            scale_tril=torch.eye(self.observed_dims)).sample((self.n_test,))

        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        if self.observed_dims == 2:
            return self.plot_2d_to_2d(output, model, metrics)


class Blobs2D(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2
        self.x_lims = [[-1.5, 1.5], [-1.5, 1.5]]

    def setup(self, stage: Optional[str] = None) -> None:
        centers = [
            [1., 0.], [-1., 0.], [0., -1.], [0., 1.],
            [1. / np.sqrt(2.), 1. / np.sqrt(2.)], [1. / np.sqrt(2.), -1. / np.sqrt(2.)],
            [-1. / np.sqrt(2.), 1. / np.sqrt(2.)], [-1. / np.sqrt(2.), -1. / np.sqrt(2.)],
        ]

        scale_tril = torch.eye(2) * FLAGS.prior_scale

        self.xs_train = None
        for c in centers:
            blob = distributions.MultivariateNormal(torch.tensor(c).float(),
                                                    scale_tril=scale_tril).sample((self.n_train // len(centers),))
            if self.xs_train is None:
                self.xs_train = blob
            else:
                self.xs_train = torch.cat((self.xs_train, blob))

        self.xs_test = None
        for c in centers:
            blob = distributions.MultivariateNormal(torch.tensor(c).float(),
                                                    scale_tril=scale_tril).sample((self.n_test // len(centers),))
            if self.xs_test is None:
                self.xs_test = blob
            else:
                self.xs_test = torch.cat((self.xs_test, blob))

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Blobs3D(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 3

        self.x_lims = [[-10, 10], [-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([5., 5., 5.]),
                                                  scale_tril=torch.eye(3)).sample((self.n_train // 3,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-5., -5., -5.]),
                                                  scale_tril=torch.eye(3)).sample((self.n_train // 3,))
        blob_3 = distributions.MultivariateNormal(loc=torch.tensor([-5., 5., -5.]),
                                                  scale_tril=torch.eye(3)).sample((self.n_train // 3,))
        self.xs_train = torch.cat((blob_1, blob_2, blob_3))

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([5., 5., 5.]),
                                                  scale_tril=torch.eye(3)).sample((self.n_test // 3,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-5., -5., -5.]),
                                                  scale_tril=torch.eye(3)).sample((self.n_test // 3,))
        blob_3 = distributions.MultivariateNormal(loc=torch.tensor([-5., 5., -5]),
                                                  scale_tril=torch.eye(3)).sample((self.n_test // 3,))
        self.xs_test = torch.cat((blob_1, blob_2, blob_3))

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        z_values_forward = output.z_values_forward.cpu().detach().numpy()
        z_values_backward = output.z_values_backward.cpu().detach().numpy()
        x_data = output.x_data.cpu().detach().numpy()
        x_prior = output.x_prior.cpu().detach().numpy()

        if z_values_backward.shape[-1] == 3:
            # Values
            z_1 = z_values_forward[:, -1]
            z_0 = z_values_backward[:, -1]

            prior_indices = ["Prior"] * x_prior.shape[0]
            prior_gen_indices = ["Prior_gen"] * z_0.shape[0]

            df_prior = pd.DataFrame(
                {
                    "Prior_1": np.concatenate((x_prior[:, 0], z_0[:, 0]), axis=0),
                    "Prior_2": np.concatenate((x_prior[:, 1], z_0[:, 1]), axis=0),
                    "Prior_3": np.concatenate((x_prior[:, 2], z_0[:, 2]), axis=0),
                    "Type": np.concatenate((prior_indices, prior_gen_indices))
                })

            data_indices = ["Data"] * x_data.shape[0]
            data_gen_indices = ["Data_gen"] * z_1.shape[0]
            df_data = pd.DataFrame(
                {
                    "Data_1": np.concatenate((x_data[:, 0], z_1[:, 0]), axis=0),
                    "Data_2": np.concatenate((x_data[:, 1], z_1[:, 1]), axis=0),
                    "Data_3": np.concatenate((x_data[:, 2], z_1[:, 2]), axis=0),
                    "Type": np.concatenate((data_indices, data_gen_indices))
                })

            # Objective
            fig_obj: Figure = figure.Figure(figsize=(15, 15))
            gs = fig_obj.add_gridspec(1, 1, height_ratios=(1,), width_ratios=(1,),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.1, hspace=0.1)
            self.plot_objective(gs, fig_obj, metrics)

            g_prior = sns.PairGrid(df_prior, hue="Type")
            g_data = sns.PairGrid(df_data, hue="Type")

            g_prior.map_upper(sns.scatterplot, alpha=.3)
            g_prior.map_lower(sns.kdeplot)
            g_prior.map_diag(sns.kdeplot)
            g_data.map_upper(sns.scatterplot, alpha=.3)
            g_data.map_lower(sns.kdeplot)
            g_data.map_diag(sns.kdeplot)

            g_prior.add_legend()
            g_data.add_legend()

            return [fig_obj, g_prior.fig, g_data.fig], ["objective", "results_prior", "results_data"]

        else:
            raise ValueError("Dims must be 3")


class DoubleWellLeft(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.x_lims = [[-1.5, 1.5], [-1.5, 1.5]]

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(
            loc=torch.tensor([-1., 0.]),
            scale_tril=torch.diag_embed(torch.tensor([0.0025, 0.01])**0.5)).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(
            loc=torch.tensor([-1., 0.]),
            scale_tril=torch.diag_embed(torch.tensor([0.0025, 0.01])**0.5)).sample((self.n_test,))
        self.xs_test = blob_1

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class DoubleWellBiModalLeft(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.x_lims = [[-1.5, 1.5], [-1.5, 1.5]]

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-1., -0.7]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_train // 2,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-1., 0.7]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_train // 2,))
        self.xs_train = torch.cat((blob_1, blob_2))

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-1., -0.7]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_test // 2,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-1., 0.7]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_test // 2,))
        self.xs_test = torch.cat((blob_1, blob_2))

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class DoubleWellRight(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.x_lims = [[-1.5, 1.5], [-1.5, 1.5]]

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(
            loc=torch.tensor([1., 0.]),
            scale_tril=torch.diag_embed(torch.tensor([0.0025, 0.01])**0.5)).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(
            loc=torch.tensor([1., 0.]),
            scale_tril=torch.diag_embed(torch.tensor([0.0025, 0.01])**0.5)).sample((self.n_test,))
        self.xs_test = blob_1

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class SpiralOne(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_test,))
        self.xs_test = blob_1

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class SpiralTwo(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-2.5, 6.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-2.5, 6.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_test,))
        self.xs_test = blob_1

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class SCurve(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        x, y = datasets.make_s_curve(self.n_train, noise=0.1)
        self.xs_train = torch.tensor(x)[:, [0, 2]]
        self.xs_train = (self.xs_train - self.xs_train.mean()) / self.xs_train.std() * self.scaling_factor
        self.xs_train = self.xs_train.float()

        # Test
        x, y = datasets.make_s_curve(self.n_test, noise=0.1)
        self.xs_test = torch.tensor(x)[:, [0, 2]]
        self.xs_test = (self.xs_test - self.xs_test.mean()) / self.xs_test.std() * self.scaling_factor
        self.xs_test = self.xs_test.float()

        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Swiss(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        x, y = datasets.make_swiss_roll(self.n_train, noise=0.2)
        self.xs_train = torch.tensor(x)[:, [0, 2]]
        self.xs_train = (self.xs_train - self.xs_train.mean()) / self.xs_train.std() * self.scaling_factor
        self.xs_train = self.xs_train.float()

        # Test
        x, y = datasets.make_swiss_roll(self.n_test, noise=0.2)
        self.xs_test = torch.tensor(x)[:, [0, 2]]
        self.xs_test = (self.xs_test - self.xs_test.mean()) / self.xs_test.std() * self.scaling_factor
        self.xs_test = self.xs_test.float()

        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Moon(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        x, y = datasets.make_moons(self.n_train, noise=0.1)
        self.xs_train = torch.tensor(x)
        self.xs_train = (self.xs_train - self.xs_train.mean()) / self.xs_train.std() * self.scaling_factor
        self.xs_train = self.xs_train.float()

        # Test
        x, y = datasets.make_moons(self.n_test, noise=0.1)
        self.xs_test = torch.tensor(x)
        self.xs_test = (self.xs_test - self.xs_test.mean()) / self.xs_test.std() * self.scaling_factor
        self.xs_test = self.xs_test.float()

        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Circle(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.scaling_factor = 5.

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        x, y = datasets.make_circles(self.n_train, factor=0.5, noise=0.05)
        self.xs_train = torch.tensor(x) * self.scaling_factor
        self.xs_train = self.xs_train.float()

        # Test
        x, y = datasets.make_circles(self.n_test, factor=0.5, noise=0.05)
        self.xs_test = torch.tensor(x) * self.scaling_factor
        self.xs_test = self.xs_test.float()

        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Checker(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 6000
        self.n_test: int = 6000
        self.observed_dims: int = 2

        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        x1 = np.random.rand(self.n_train) * 4 - 2
        x2_ = np.random.rand(self.n_train) - np.random.randint(0, 2, self.n_train) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * self.scaling_factor
        self.xs_train = torch.from_numpy(x).float()

        # x_max = np.max(x[:, 0])
        # x_min = np.min(x[:, 0])
        # y_max = np.max(x[:, 1])
        # y_min = np.min(x[:, 1])

        # Test
        x1 = np.random.rand(self.n_test) * 4 - 2
        x2_ = np.random.rand(self.n_test) - np.random.randint(0, 2, self.n_test) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * self.scaling_factor
        self.xs_test = torch.from_numpy(x).float()

        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class GaussianBoundLeft(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 5000
        self.n_test: int = 5000
        self.observed_dims: int = 2

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(
            loc=torch.tensor([-2.] * self.observed_dims),
            scale_tril=0.3 * torch.eye(self.observed_dims)).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-2.] * self.observed_dims),
                                                  scale_tril=0.3 * torch.eye(self.observed_dims)).sample((self.n_test,))
        self.xs_test = blob_1

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        z_values_backward = output.z_values_backward.cpu().detach().numpy()

        if z_values_backward.shape[-1] == self.observed_dims:

            fig_obj: Figure = figure.Figure(figsize=(15, 15))
            gs = fig_obj.add_gridspec(1, 1, height_ratios=(1,), width_ratios=(1,),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.1, hspace=0.1)

            self.plot_objective(gs, fig_obj, metrics)
            mean_prior = np.mean(metrics.mean_prior.cpu().detach().numpy(), -1)
            mean_data = np.mean(metrics.mean_data.cpu().detach().numpy(), -1)
            std_prior = np.mean(metrics.std_prior.cpu().detach().numpy(), -1)
            std_data = np.mean(metrics.std_data.cpu().detach().numpy(), -1)

            epochs_array = np.arange(0, mean_prior.shape[0])
            indices = np.arange(0, mean_prior.shape[0])

            fig: Figure = figure.Figure(figsize=(15, 15))
            ax: Axes = fig.add_subplot(1, 1, 1)

            if len(indices) > 1:
                ax.hlines(2, indices[0], indices[-1], color="k", linestyles="solid")
                ax.hlines(-2, indices[0], indices[-1], color="k", linestyles="solid")
                ax.hlines(0.5, indices[0], indices[-1], color="0.4", linestyles="solid")
                ax.hlines(0.3, indices[0], indices[-1], color="0.4", linestyles="solid")

            ax.plot(epochs_array[indices], mean_prior[indices], c="r", label="Prior mean", linestyle="dashed")
            ax.plot(epochs_array[indices], mean_data[indices], c="b", label="Data mean", linestyle="dashed")
            ax.plot(epochs_array[indices], std_prior[indices], c="r", label="Prior std", linestyle="dotted")
            ax.plot(epochs_array[indices], std_data[indices], c="b", label="Data std", linestyle="dotted")
            ax.set_xlabel("Epoch")
            ax.legend(loc=2)
            ax.grid(True)

            return [fig_obj, fig], \
                   ["objective", "means_stds"]
            # return [fig], \
            #        ["means_stds"]

        else:
            raise ValueError(f"Dims must be {self.observed_dims}")


class GaussianBoundRight(GaussianBoundLeft):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 5000
        self.n_test: int = 5000
        self.observed_dims: int = 2

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([2.] * self.observed_dims),
                                                  scale_tril=0.5*torch.eye(self.observed_dims)).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([2.] * self.observed_dims),
                                                  scale_tril=0.5*torch.eye(self.observed_dims)).sample((self.n_test,))
        self.xs_test = blob_1

        if self.prior_dataset is not None:
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()


class MNIST(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 60000
        self.n_test: int = 10000
        self.observed_dims: int = 784

        self.num_images_eval = 64

        self.transform = torch.nn.Sequential(transforms.Normalize((0.1307,), (0.3081,)))

    def prepare_data(self):
        vision_datasets.MNIST("datasets/", True, download=True)
        vision_datasets.MNIST("datasets/", False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Train
        images_train = vision_datasets.MNIST("datasets/", True)

        indices = torch.randperm(60000)[:self.n_train]
        images_train = self.transform(images_train.data.float())[indices]
        self.xs_train = images_train.reshape(-1, self.observed_dims)

        # Test
        images_test = vision_datasets.MNIST("datasets/", False)
        indices = torch.randperm(10000)[:self.n_test]
        # self.test_labels = torch.tensor(images_test.targets)[indices]
        images_test = self.transform(images_test.data.float())[indices]
        self.xs_test = images_test.reshape(-1, self.observed_dims)

        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.observed_dims = self.observed_dims
            self.prior_dataset.setup(stage)
            self.calculate_max_diffusion()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        z_values_forward = output.z_values_forward.cpu().detach()
        x_gen = z_values_forward[-1]

        # fig_obj: Figure = figure.Figure(figsize=(15, 15))
        # gs = fig_obj.add_gridspec(1, 1, height_ratios=(1,), width_ratios=(1,),
        #                           left=0.1, right=0.9, bottom=0.1, top=0.9,
        #                           wspace=0.1, hspace=0.1)

        # self.plot_objective(gs, fig_obj, metrics)

        indices = np.random.choice(x_gen.shape[0], self.num_images_eval)
        images = torch.tile((x_gen[indices].reshape(-1, 28, 28)).unsqueeze(1), (1, 3, 1, 1))

        grid_image = utils.make_grid(images, nrow=int(np.sqrt(self.num_images_eval)))
        grid_image = grid_image.permute(1, 2, 0)[..., 0]

        fig: Figure = figure.Figure(figsize=(15, 15))
        ax: Axes = fig.add_subplot(1, 1, 1)

        ax.imshow(grid_image.numpy())

        # return [fig_obj, fig], \
        #        ["objective", "samples"]
        return [fig], \
               ["samples"]


datasets_dict = {
    "gaussian": Gaussian,
    # Experiments
    "blobs_2d": Blobs2D,
    "blobs_3d": Blobs3D,
    "double_well_left": DoubleWellLeft,
    "double_well_bimodal_left": DoubleWellBiModalLeft,
    "double_well_right": DoubleWellRight,
    "spiral_one": SpiralOne,
    "spiral_two": SpiralTwo,
    "s_curve": SCurve,
    "swiss_roll": Swiss,
    "moon": Moon,
    "circle": Circle,
    "checker": Checker,
    # Bounds experiment
    "gaussian_bound_left": GaussianBoundLeft,
    "gaussian_bound_right": GaussianBoundRight,
    # MNIST,
    "mnist": MNIST,
}
