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

from models import Output, Model, Metrics

Tensor = torch.Tensor

datasets_list = [
                    # Priors
                    "gaussian",
                    # Experiments
                    "toy_experiment_blobs_2d",
                    "toy_experiment_blobs_3d",
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
                    self.prior_dataset.xs_train, self.batch_size, shuffle=True, num_workers=4),
                "data": DataLoader(self.xs_train, self.batch_size, shuffle=True, num_workers=4)}
            return CombinedLoader(loaders, "max_size_cycle")
        else:
            return DataLoader(self.xs_train, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        if self.prior_dataset is not None:
            loaders = {
                "prior": DataLoader(self.prior_dataset.xs_test, self.batch_size, num_workers=4),
                "data": DataLoader(self.xs_test, self.batch_size, num_workers=4)}
            return CombinedLoader(loaders, "max_size_cycle")
        else:
            return DataLoader(self.xs_test, self.batch_size, num_workers=4)

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

            fig: Figure = figure.Figure(figsize=(15, 15))
            gs = fig.add_gridspec(3, 2, height_ratios=(2, 7, 7), width_ratios=(1, 1),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)

            self.plot_objective(gs, fig, metrics)
            # Plotting
            ax_z_backwards: Axes = fig.add_subplot(gs[1, 0])
            ax_z_backwards.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_backwards.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            ax_z_forwards: Axes = fig.add_subplot(gs[2, 0])
            ax_z_forwards.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_forwards.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_z.axis("off")

            ax_z_0: Axes = fig.add_subplot(gs[1, 1])
            ax_z_0.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_0.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_x.axis("off")

            ax_z_1: Axes = fig.add_subplot(gs[2, 1])
            ax_z_1.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z_1.set_xlim(self.x_lims[0][0], self.x_lims[0][1])

            t_values = model.time_values.cpu().detach().numpy()

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

            return [fig], ["results"]

        else:
            raise ValueError("Dims must be 2")


class Gaussian(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 500
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
        self.n_train: int = 500
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
        self.n_train: int = 500
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

            df_prior = pd.DataFrame(
                {
                    "Prior_1": x_prior[:, 0], "Prior_gen_1": z_0[:, 0],
                    "Prior_2": x_prior[:, 1], "Prior_gen_2": z_0[:, 1],
                    "Prior_3": x_prior[:, 2], "Prior_gen_3": z_0[:, 2],
                })
            df_data = pd.DataFrame(
                {
                    "Data_1": x_data[:, 0], "Data_gen_1": z_1[:, 0],
                    "Data_2": x_data[:, 1], "Data_gen_2": z_1[:, 1],
                    "Data_3": x_data[:, 2], "Data_gen_3": z_1[:, 2],
                })

            # Objective
            fig_obj: Figure = figure.Figure(figsize=(15, 15))
            gs = fig_obj.add_gridspec(1, 1, height_ratios=(1,), width_ratios=(1,),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.1, hspace=0.1)
            self.plot_objective(gs, fig_obj, metrics)

            g_prior = sns.PairGrid(df_prior)
            g_data = sns.PairGrid(df_data)

            g_prior.map_upper(sns.scatterplot)
            g_prior.map_lower(sns.kdeplot)
            g_prior.map_diag(sns.kdeplot)
            g_data.map_upper(sns.scatterplot)
            g_data.map_lower(sns.kdeplot)
            g_data.map_diag(sns.kdeplot)

            return [fig_obj, g_prior.fig, g_data.fig], ["objective", "results_prior", "results_data"]

        else:
            raise ValueError("Dims must be 3")


datasets_dict = {
    "gaussian": Gaussian,
    # Experiments
    "toy_experiment_blobs_2d": Blobs2D,
    "toy_experiment_blobs_3d": Blobs3D,
}
