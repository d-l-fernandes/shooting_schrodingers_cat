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
from typing import Optional, List, Tuple, Any
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn import datasets

from models import Output, Model, Metrics
from sde import prior_sdes

Tensor = torch.Tensor

datasets_list = [
                    # Priors
                    "gaussian",
                    # Experiments
                    "toy_experiment_blobs_2d",
                    "toy_experiment_blobs_3d",
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
]

flags.DEFINE_integer("batch_size", 10, "Batch Size.")
flags.DEFINE_integer("dims", 2, "Number of dims.")
flags.DEFINE_enum("prior", "gaussian", datasets_list, "Prior to use.")
flags.DEFINE_enum("dataset", "toy_experiment_blobs_2d", datasets_list, "Dataset to use.")

FLAGS = flags.FLAGS


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
        self.x_lims: Optional[List[List[float]]] = None

        # Train arrays
        self.xs_train: Tensor = torch.ones((self.n_train, self.observed_dims))
        # Test arrays
        self.xs_test: Tensor = torch.ones((self.n_test, self.observed_dims))

        torch.manual_seed(42)
        np.random.seed(42)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train arrays
        self.xs_train: Tensor = torch.ones((self.n_train, self.observed_dims))
        # Test arrays
        self.xs_test: Tensor = torch.ones((self.n_test, self.observed_dims))

    def train_dataloader(self):
        if self.prior_dataset is not None:
            loaders = {
                "prior": DataLoader(
                    self.prior_dataset.xs_train, self.batch_size, shuffle=True, num_workers=0, pin_memory=True),
                "data": DataLoader(self.xs_train, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)}
            return CombinedLoader(loaders, "max_size_cycle")
        else:
            return DataLoader(self.xs_train, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        if self.prior_dataset is not None:
            loaders = {
                "prior": DataLoader(self.prior_dataset.xs_test, self.batch_size, num_workers=0, pin_memory=True),
                "data": DataLoader(self.xs_test, self.batch_size, num_workers=0, pin_memory=True)}
            return CombinedLoader(loaders, "max_size_cycle")
        else:
            return DataLoader(self.xs_test, self.batch_size, num_workers=0, pin_memory=True)

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        raise NotImplementedError

    @staticmethod
    def plot_objective(gs: Any, fig: Figure, metrics: Metrics):
        wasserstein_prior = metrics.wasserstein_prior.cpu().detach().numpy()
        wasserstein_data = metrics.wasserstein_data.cpu().detach().numpy()
        wasserstein_total = metrics.wasserstein_total.cpu().detach().numpy()

        # Objective plotting
        epochs_array = np.arange(0, wasserstein_prior.shape[0])
        ax_objective: Axes = fig.add_subplot(gs[0, :])

        indices = np.arange(0, wasserstein_prior.shape[0])
        ax_objective.plot(epochs_array[indices], wasserstein_prior[indices], c="r", label="prior")
        ax_objective.plot(epochs_array[indices], wasserstein_data[indices], c="b", label="data")
        ax_objective.plot(epochs_array[indices], wasserstein_total[indices], c="k", label="total")
        ax_objective.set_xlabel("Epoch")
        ax_objective.set_yscale("symlog")
        ax_objective.legend(loc=2)
        ax_objective.grid(True)
        if len(indices) != 0:
            total_min = np.argmin(wasserstein_total)
            ax_objective.vlines(total_min, wasserstein_total[total_min], 2 * wasserstein_total[total_min], color="k")
            ax_objective.annotate(f"{wasserstein_total[total_min]:.3f}",
                                  xy=(total_min, 2 * wasserstein_total[total_min]),
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

            self.plot_objective(gs, fig_obj, metrics)
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

            t_values = model.time_values.cpu().detach().numpy()

            if type(model.prior_sde) in [prior_sdes.Hill, prior_sdes.Maze, prior_sdes.Spiral]:
                xx, yy = np.meshgrid(np.linspace(self.x_lims[0][0], self.x_lims[0][1], 100),
                                     np.linspace(self.x_lims[1][0], self.x_lims[1][1], 100))
                zz = model.prior_sde.u(
                    torch.tensor(xx).to(output.z_values_forward.device),
                    torch.tensor(yy).to(output.z_values_forward.device),
                ).detach().cpu().numpy()
                ax_z_1.pcolormesh(xx, yy, zz, alpha=0.6, cmap="OrRd", shading="auto")
                ax_z_0.pcolormesh(xx, yy, zz, alpha=0.6, cmap="OrRd", shading="auto")
                ax_z_backwards.pcolormesh(xx, yy, zz, alpha=0.6, cmap="OrRd", shading="auto")
                ax_z_forwards.pcolormesh(xx, yy, zz, alpha=0.6, cmap="OrRd", shading="auto")

            norm = plt.Normalize(t_values.min(), t_values.max())
            ax_z_backwards.scatter(z_values_backward[:, -1, 0], z_values_backward[:, -1, 1],
                                   c=np.ones_like(z_values_backward[:, -1, 0]), cmap="gnuplot", alpha=0.5, marker="x")
            ax_z_forwards.scatter(z_values_forward[:, -1, 0], z_values_forward[:, -1, 1],
                                  c=np.ones_like(z_values_forward[:, -1, 0]), cmap="gnuplot", alpha=0.5, marker="x")
            ax_z_1.scatter(x_data[:, 0], x_data[:, 1], c="r", marker='1')
            ax_z_1.scatter(z_1[:, 0], z_1[:, 1], c="b", marker='*', alpha=0.5)
            ax_z_0.scatter(x_prior[:, 0], x_prior[:, 1], c="r", marker='1')
            ax_z_0.scatter(z_0[:, 0], z_0[:, 1], c="b", marker='*', alpha=0.5)
            for i in range(z_values_forward.shape[0]):
                points = np.array([z_values_forward[i, :, 0], z_values_forward[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.5)
                line = ax_z_forwards.add_collection(lc)
                del lc
            for i in range(z_values_backward.shape[0]):
                points = np.array([z_values_backward[i, :, 0], z_values_backward[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.5)
                line = ax_z_backwards.add_collection(lc)
                del lc

            return [fig_obj, fig_z_forwards, fig_z_backwards, fig_z_0, fig_z_1], \
                   ["objective", "forwards", "backwards", "z_0", "z_1"]

        else:
            raise ValueError("Dims must be 2")


class Gaussian(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = FLAGS.dims

        self.x_lims = [[-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        self.xs_train = distributions.MultivariateNormal(
            loc=torch.tensor([0.] * self.observed_dims),
            scale_tril=torch.eye(self.observed_dims)).sample((self.n_train,))
        # Test
        self.xs_test = distributions.MultivariateNormal(
            loc=torch.tensor([0.] * self.observed_dims),
            scale_tril=torch.eye(self.observed_dims)).sample((self.n_test,))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        if self.observed_dims == 2:
            return self.plot_2d_to_2d(output, model, metrics)


class Blobs2D(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([5., 5.]),
                                                  scale_tril=torch.eye(2)).sample((self.n_train // 3,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-5., -5.]),
                                                  scale_tril=torch.eye(2)).sample((self.n_train // 3,))
        blob_3 = distributions.MultivariateNormal(loc=torch.tensor([-5., 5.]),
                                                  scale_tril=torch.eye(2)).sample((self.n_train // 3,))
        self.xs_train = torch.cat((blob_1, blob_2, blob_3))

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([5., 5.]),
                                                  scale_tril=torch.eye(2)).sample((self.n_test // 3,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-5., -5.]),
                                                  scale_tril=torch.eye(2)).sample((self.n_test // 3,))
        blob_3 = distributions.MultivariateNormal(loc=torch.tensor([-5., 5.]),
                                                  scale_tril=torch.eye(2)).sample((self.n_test // 3,))
        self.xs_test = torch.cat((blob_1, blob_2, blob_3))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Blobs3D(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 3

        self.x_lims = [[-10, 10], [-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
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
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-1.5, 1.5], [-1.5, 1.5]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-1., 0.]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-1., 0.]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_test,))
        self.xs_test = blob_1

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class DoubleWellBiModalLeft(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-1.5, 1.5], [-1.5, 1.5]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
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

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class DoubleWellRight(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-1.5, 1.5], [-1.5, 1.5]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([1., 0.]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([1., 0.]),
                                                  scale_tril=torch.eye(2) * 0.1).sample((self.n_test,))
        self.xs_test = blob_1

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class SpiralOne(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_test,))
        self.xs_test = blob_1

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class SpiralTwo(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-2.5, 6.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_train,))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([-2.5, 6.]),
                                                  scale_tril=torch.eye(2) * 0.5).sample((self.n_test,))
        self.xs_test = blob_1

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class SCurve(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]
        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.setup(stage)
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

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Swiss(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]
        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.setup(stage)
        # Train
        x, y = datasets.make_swiss_roll(self.n_train, noise=0.1)
        self.xs_train = torch.tensor(x)[:, [0, 2]]
        self.xs_train = (self.xs_train - self.xs_train.mean()) / self.xs_train.std() * self.scaling_factor
        self.xs_train = self.xs_train.float()

        # Test
        x, y = datasets.make_swiss_roll(self.n_test, noise=0.1)
        self.xs_test = torch.tensor(x)[:, [0, 2]]
        self.xs_test = (self.xs_test - self.xs_test.mean()) / self.xs_test.std() * self.scaling_factor
        self.xs_test = self.xs_test.float()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Moon(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]
        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.setup(stage)
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

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Circle(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]
        self.scaling_factor = 7.

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.setup(stage)
        # Train
        x, y = datasets.make_circles(self.n_train, factor=0.5, noise=0.1)
        self.xs_train = torch.tensor(x) * self.scaling_factor
        self.xs_train = self.xs_train.float()

        # Test
        x, y = datasets.make_circles(self.n_test, factor=0.5, noise=0.1)
        self.xs_test = torch.tensor(x) * self.scaling_factor
        self.xs_test = self.xs_test.float()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Checker(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 3000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]
        self.scaling_factor = 4.

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.n_train = self.n_train
            self.prior_dataset.n_test = self.n_test
            self.prior_dataset.setup(stage)
        # Train
        x1 = np.random.rand(self.n_train) * 4 - 2
        x2_ = np.random.rand(self.n_train) - np.random.randint(0, 2, self.n_train) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * 7.5
        self.xs_train = torch.from_numpy(x).float()

        x_max = np.max(x[:, 0])
        x_min = np.min(x[:, 0])
        y_max = np.max(x[:, 1])
        y_min = np.min(x[:, 1])
        self.x_lims = [[x_min-1, x_max+1], [y_min-1, y_max+1]]

        # Test
        x1 = np.random.rand(self.n_test) * 4 - 2
        x2_ = np.random.rand(self.n_test) - np.random.randint(0, 2, self.n_test) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * 7.5
        self.xs_test = torch.from_numpy(x).float()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


datasets_dict = {
    "gaussian": Gaussian,
    # Experiments
    "toy_experiment_blobs_2d": Blobs2D,
    "toy_experiment_blobs_3d": Blobs3D,
    "double_well_left": DoubleWellLeft,
    "double_well_bimodal_left": DoubleWellBiModalLeft,
    "double_well_right": DoubleWellRight,
    "spiral_one": SpiralOne,
    "spiral_two": SpiralTwo,
    "s_curve": SCurve,
    "swiss_roll": Swiss,
    "moon": Moon,
    "circle": Circle,
    "checker": Checker
}
