import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as skdatasets
import torch
import torch.distributions as distributions
from absl import flags
from matplotlib import figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, List, Tuple, Any
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

from models import Output, Model, Metrics
from pretrained.mnist_vae import vae

Tensor = torch.Tensor

datasets_list = [
                    # Priors
                    "gaussian",
                    # Experiments
                    "toy_experiment_blobs_2d",
                    "toy_experiment_circle",
                    "toy_experiment_circle_and_blob_concentric",
                    "toy_experiment_circle_and_blob",
                    "toy_experiment_banana",
                    "toy_experiment_dual_moon",
                    "toy_experiment_uneven_circle",
                    "mnist_2d",
                    "gaussian_50d"
                ]

flags.DEFINE_integer("batch_size", 10, "Batch Size.")
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
        z_generated = output.z_generated.cpu().detach().numpy()
        x_data = output.x_data.cpu().detach().numpy()

        if z_values_backward.shape[-1] == 2:
            # Values
            x_gen = z_values_forward[:, -1]

            fig: Figure = figure.Figure(figsize=(15, 15))
            gs = fig.add_gridspec(3, 2, height_ratios=(2, 7, 7), width_ratios=(1, 1),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)

            self.plot_objective(gs, fig, metrics)
            # Plotting
            ax_z1: Axes = fig.add_subplot(gs[1, 0])
            ax_z1.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z1.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            ax_z2: Axes = fig.add_subplot(gs[2, 0])
            ax_z2.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z2.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_z.axis("off")

            ax_x: Axes = fig.add_subplot(gs[1, 1])
            ax_x.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_x.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_x.axis("off")

            ax_generative: Axes = fig.add_subplot(gs[2, 1])
            ax_generative.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_generative.set_xlim(self.x_lims[0][0], self.x_lims[0][1])

            t_values = model.time_values.cpu().detach().numpy()

            norm = plt.Normalize(t_values.min(), t_values.max())
            ax_z1.scatter(z_values_backward[:, -1, 0], z_values_backward[:, -1, 1],
                          c=np.ones_like(z_values_backward[:, -1, 0]), cmap="gnuplot", alpha=0.5, marker="x")
            ax_z2.scatter(z_values_forward[:, -1, 0], z_values_forward[:, -1, 1],
                          c=np.ones_like(z_values_forward[:, -1, 0]), cmap="gnuplot", alpha=0.5, marker="x")
            ax_x.scatter(x_data[:, 0], x_data[:, 1], c="r", marker='1')
            ax_x.scatter(x_gen[:, 0], x_gen[:, 1], c="b", marker='*')
            for i in range(z_values_forward.shape[0]):
                points = np.array([z_values_backward[i, :, 0], z_values_backward[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.5)
                line = ax_z1.add_collection(lc)
                del lc

                points = np.array([z_values_forward[i, :, 0], z_values_forward[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.5)
                line = ax_z2.add_collection(lc)
                del lc

                # Generative plot
                points = np.array(
                    [z_generated[i, :, 0], z_generated[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='copper', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.7)
                line = ax_generative.add_collection(lc)
                del lc
            ax_generative.scatter(x_data[:, 0], x_data[:, 1], c="red", marker='1')

            return [fig], ["results"]

        else:
            raise ValueError("Latent dims must be 2")


class Gaussian(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 1000
        self.observed_dims: int = 2

        self.x_lims = [[-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        self.xs_train = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                         scale_tril=torch.eye(2)).sample((self.n_train,))
        # Test
        self.xs_test = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                        scale_tril=torch.eye(2)).sample((self.n_test,))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Blobs2D(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 1000
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


class One1DBlob(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 100
        self.n_test: int = 100
        self.observed_dims: int = 1

        self.x_lims = [[-2, 2]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([0.]),
                                                  scale_tril=torch.eye(1) * 0.5).sample((self.n_train, ))
        self.xs_train = blob_1

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([0.]),
                                                  scale_tril=torch.eye(1) * 0.5).sample((self.n_test, ))
        self.xs_test = blob_1

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Two1DBlobs(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 100
        self.n_test: int = 100
        self.observed_dims: int = 1

        self.x_lims = [[-2, 2]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([1.8]),
                                                  scale_tril=torch.eye(1) * 0.6).sample((self.n_train // 2,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-1.9]),
                                                  scale_tril=torch.eye(1) * 0.6).sample((self.n_train // 2,))
        self.xs_train = torch.cat((blob_1, blob_2))

        # Test
        blob_1 = distributions.MultivariateNormal(loc=torch.tensor([1.8]),
                                                  scale_tril=torch.eye(1) * 0.6).sample((self.n_test // 2,))
        blob_2 = distributions.MultivariateNormal(loc=torch.tensor([-1.9]),
                                                  scale_tril=torch.eye(1) * 0.6).sample((self.n_test // 2,))
        self.xs_test = torch.cat((blob_1, blob_2))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        z_values_forward = output.z_values_forward.cpu().detach().numpy()
        z_values_backward = output.z_values_backward.cpu().detach().numpy()
        z_generated = output.z_generated.cpu().detach().numpy()
        x_data = output.x_data.cpu().detach().numpy()

        if z_values_backward.shape[-1] == 2:
            # Values
            x_gen = z_values_forward[:, -1]

            fig: Figure = figure.Figure(figsize=(15, 15))
            gs = fig.add_gridspec(3, 2, height_ratios=(2, 7, 7), width_ratios=(1, 1),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)

            self.plot_objective(gs, fig, metrics)
            # Plotting
            ax_z1: Axes = fig.add_subplot(gs[1, 0])
            ax_z1.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z1.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            ax_z2: Axes = fig.add_subplot(gs[2, 0])
            ax_z2.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_z2.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_z.axis("off")

            ax_x: Axes = fig.add_subplot(gs[1, 1])
            ax_x.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_x.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            # ax_x.axis("off")

            ax_generative: Axes = fig.add_subplot(gs[2, 1])
            ax_generative.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_generative.set_xlim(self.x_lims[0][0], self.x_lims[0][1])

            t_values = model.time_values.cpu().detach().numpy()

            norm = plt.Normalize(t_values.min(), t_values.max())
            ax_z1.scatter(z_values_backward[:, -1, 0], z_values_backward[:, -1, 1],
                          c=np.ones_like(z_values_backward[:, -1, 0]), cmap="gnuplot", alpha=0.5, marker="x")
            ax_z2.scatter(z_values_forward[:, -1, 0], z_values_forward[:, -1, 1],
                          c=np.ones_like(z_values_forward[:, -1, 0]), cmap="gnuplot", alpha=0.5, marker="x")
            ax_x.scatter(x_data[:, 0], x_data[:, 1], c="r", marker='1')
            ax_x.scatter(x_gen[:, 0], x_gen[:, 1], c="b", marker='*')
            for i in range(z_values_forward.shape[0]):
                points = np.array([z_values_backward[i, :, 0], z_values_backward[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.5)
                line = ax_z1.add_collection(lc)
                del lc

                points = np.array([z_values_forward[i, :, 0], z_values_forward[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='gnuplot', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.5)
                line = ax_z2.add_collection(lc)
                del lc

                # Generative plot
                points = np.array(
                    [z_generated[i, :, 0], z_generated[i, :, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='copper', norm=norm)
                lc.set_array(t_values)
                lc.set_linewidth(2)
                lc.set_alpha(0.7)
                line = ax_generative.add_collection(lc)
                del lc
            ax_generative.scatter(x_data[:, 0], x_data[:, 1], c="red", marker='1')

            return [fig], ["results"]

        else:
            raise ValueError("Latent dims must be 2")


class Circle(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 300
        self.observed_dims: int = 2

        self.x_lims = [[-8, 8], [-8, 8]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        angle = 2 * np.pi * torch.rand((self.n_train,))
        mean = torch.cat((5 * torch.cos(angle)[:, None], 5 * torch.sin(angle)[:, None]), 1)
        self.xs_train = distributions.MultivariateNormal(
            loc=mean, scale_tril=torch.eye(2) * 0.3).sample()

        # Test
        angle = 2 * np.pi * torch.rand((self.n_test,))
        mean = torch.cat((5 * torch.cos(angle)[:, None], 5 * torch.sin(angle)[:, None]), 1)
        self.xs_test = distributions.MultivariateNormal(
            loc=mean, scale_tril=torch.eye(2) * 0.3).sample()

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class CircleAndBlobConcentric(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 300
        self.observed_dims: int = 2

        self.x_lims = [[-8, 8], [-8, 8]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                scale_tril=torch.eye(2)).sample((self.n_train // 2,))
        angle = 2 * np.pi * torch.rand((self.n_train // 2,))
        mean = torch.cat((5 * torch.cos(angle)[:, None], 5 * torch.sin(angle)[:, None]), 1)
        circle = distributions.MultivariateNormal(loc=mean, scale_tril=torch.eye(2) * 0.3).sample()
        self.xs_train = torch.cat((blob, circle))

        # Test
        blob = distributions.MultivariateNormal(loc=torch.tensor([0., 0.]),
                                                scale_tril=torch.eye(2)).sample((self.n_train // 2,))
        angle = 2 * np.pi * torch.rand((self.n_train // 2,))
        mean = torch.cat((5 * torch.cos(angle)[:, None], 5 * torch.sin(angle)[:, None]), 1)
        circle = distributions.MultivariateNormal(loc=mean, scale_tril=torch.eye(2) * 0.3).sample()
        self.xs_test = torch.cat((blob, circle))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class CircleAndBlob(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 300
        self.observed_dims: int = 2

        self.x_lims = [[-15, 10], [-15, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob = distributions.MultivariateNormal(loc=torch.tensor([5., 5.]),
                                                scale_tril=torch.eye(2)).sample((self.n_train // 2,))
        angle = 2 * np.pi * torch.rand((self.n_train // 2,))
        mean = torch.cat((5 * torch.cos(angle)[:, None] - 8, 5 * torch.sin(angle)[:, None] - 8), 1)
        circle = distributions.MultivariateNormal(loc=mean, scale_tril=torch.eye(2) * 0.3).sample()
        self.xs_train = torch.cat((blob, circle))

        # Test
        blob = distributions.MultivariateNormal(loc=torch.tensor([5., 5.]),
                                                scale_tril=torch.eye(2)).sample((self.n_train // 2,))
        angle = 2 * np.pi * torch.rand((self.n_train // 2,))
        mean = torch.cat((5 * torch.cos(angle)[:, None] - 8, 5 * torch.sin(angle)[:, None] - 8), 1)
        circle = distributions.MultivariateNormal(loc=mean, scale_tril=torch.eye(2) * 0.3).sample()
        self.xs_test = torch.cat((blob, circle))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class Banana(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 400
        self.n_test: int = 1000
        self.observed_dims: int = 2

        self.x_lims = [[-3, 3], [-5, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        blob = distributions.MultivariateNormal(
            loc=torch.tensor([0., 0.]), scale_tril=torch.tensor([[1., 0.], [0., 0.3]])).sample((self.n_train,))
        blob[:, 1] = blob[:, 0]**2 + blob[:, 1]
        self.xs_train = blob

        # Test
        blob = distributions.MultivariateNormal(
            loc=torch.tensor([0., 0.]), scale_tril=torch.tensor([[1., 0.], [0., 0.3]])).sample((self.n_train,))
        blob[:, 1] = blob[:, 0]**2 + blob[:, 1]
        self.xs_test = blob

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class DualMoon(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 300
        self.observed_dims: int = 2

        self.x_lims = [[-8, 9], [-6, 6]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        self.xs_train, _ = skdatasets.make_moons(self.n_train, noise=0.1)
        self.xs_train = 5 * self.xs_train
        self.xs_train[:, 0] = self.xs_train[:, 0] - 2
        self.xs_train[:, 1] = self.xs_train[:, 1] - 1
        self.xs_train = torch.tensor(self.xs_train, dtype=torch.float32)

        # Test
        self.xs_test, _ = skdatasets.make_moons(self.n_test, noise=0.1)
        self.xs_test = 5 * self.xs_test
        self.xs_test[:, 0] = self.xs_test[:, 0] - 2
        self.xs_test[:, 1] = self.xs_test[:, 1] - 1
        self.xs_test = torch.tensor(self.xs_test, dtype=torch.float32)

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class UnevenCircle(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 300
        self.observed_dims: int = 2

        self.x_lims = [[-8, 8], [-8, 8]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.setup(stage)
        # Train
        angle_top = np.pi * torch.normal(0.5, 0.2, size=(self.n_train // 2,))
        angle_bottom = np.pi * torch.normal(-0.5, 0.2, size=(self.n_train // 2,))
        self.xs_train = torch.cat(
            (torch.cat((5*torch.cos(angle_top)[:, None], 5*torch.sin(angle_top)[:, None]), 1),
             torch.cat((5*torch.cos(angle_bottom)[:, None], 5*torch.sin(angle_bottom)[:, None]), 1)),
            0
        ) + torch.normal(0, 0.5, (self.n_train, 2))

        # Test
        angle_top = np.pi * torch.normal(0.5, 0.2, size=(self.n_test // 2,))
        angle_bottom = np.pi * torch.normal(-0.5, 0.2, size=(self.n_test // 2,))
        self.xs_test = torch.cat(
            (torch.cat((5*torch.cos(angle_top)[:, None], 5*torch.sin(angle_top)[:, None]), 1),
             torch.cat((5*torch.cos(angle_bottom)[:, None], 5*torch.sin(angle_bottom)[:, None]), 1)),
            0
        ) + torch.normal(0, 0.5, (self.n_test, 2))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        return self.plot_2d_to_2d(output, model, metrics)


class MNIST2D(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 10000
        self.n_test: int = 3000
        self.observed_dims: int = 2

        self.x_lims = [[-4, 4], [-4, 4]]
        self.x_lims_gen = [[-2, 2], [-2, 2]]

        self.net = vae.VAE()
        self.net.load_state_dict(torch.load("pretrained/mnist_vae/weights/vae_epoch_100.pth"))

        self.transform = torch.nn.Sequential(transforms.Normalize((0.1307,), (0.3081,)))

    def prepare_data(self):
        datasets.MNIST("pretrained/mnist_vae/data/", True)
        datasets.MNIST("pretrained/mnist_vae/data/", False)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            self.prior_dataset.n_train = 1000
            self.prior_dataset.setup(stage)
        # Train
        images_train = datasets.MNIST("pretrained/mnist_vae/data/", True)

        indices = torch.randperm(60000)[:self.n_train]
        images_train = self.transform(images_train.data.float())[indices]
        with torch.no_grad():
            self.xs_train, _ = self.net.encode(images_train.reshape(-1, 784))

        # Test
        images_test = datasets.MNIST("pretrained/mnist_vae/data/", False)
        indices = torch.randperm(10000)[:self.n_test]
        # self.test_labels = torch.tensor(images_test.targets)[indices]
        images_test = self.transform(images_test.data.float())[indices]
        with torch.no_grad():
            self.xs_test, _ = self.net.encode(images_test.reshape(-1, 784))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        z_values_forward = output.z_values_forward.cpu().detach().numpy()
        z_values_backward = output.z_values_backward.cpu().detach().numpy()
        z_generated = output.z_generated.cpu().detach().numpy()
        x_data = output.x_data.cpu().detach().numpy()

        if z_values_backward.shape[-1] == 2:
            # Values
            x_gen = z_values_forward[:, -1]

            fig: Figure = figure.Figure(figsize=(15, 15))
            gs = fig.add_gridspec(5, 2, height_ratios=(2, 7, 7, 7, 7), width_ratios=(1, 1),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)

            self.plot_objective(gs, fig, metrics)
            # Plotting
            ax_data: Axes = fig.add_subplot(gs[1, 0])
            ax_data.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_data.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            ax_data.scatter(x_data[:, 0], x_data[:, 1],
                            # c=np.array(self.test_labels), cmap=plt.get_cmap("tab10"),
                            alpha=0.6, s=5.)

            ax_reco: Axes = fig.add_subplot(gs[1, 1])
            ax_reco.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_reco.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            ax_reco.scatter(x_gen[:, 0], x_gen[:, 1],
                            c="r",
                            # c=np.array(self.test_labels), cmap=plt.get_cmap("tab10"),
                            alpha=0.6, s=5.)

            ax_prior: Axes = fig.add_subplot(gs[2, 0])
            ax_prior.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_prior.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            ax_prior.scatter(z_values_forward[:, 0, 0],
                             z_values_forward[:, 0, 1],
                             # c=np.array(self.test_labels), cmap=plt.get_cmap("tab10"),
                             alpha=0.6, s=5.)
            ax_orign: Axes = fig.add_subplot(gs[2, 1])
            ax_orign.set_ylim(self.x_lims[1][0], self.x_lims[1][1])
            ax_orign.set_xlim(self.x_lims[0][0], self.x_lims[0][1])
            ax_orign.scatter(z_generated[:, -1, 0],
                             z_generated[:, -1, 1],
                             c="g",
                             # c=np.array(self.test_labels), cmap=plt.get_cmap("tab10"),
                             alpha=0.6, s=5.)

            # Generative plot
            ax_generative: Axes = fig.add_subplot(gs[3:, 0:])
            ax_generative.set_ylim(self.x_lims_gen[1][0], self.x_lims_gen[1][1])
            ax_generative.set_xlim(self.x_lims_gen[0][0], self.x_lims_gen[0][1])

            num_points_x = 26
            num_points_y = 16

            xx, yy = torch.meshgrid(
                torch.linspace(self.x_lims_gen[0][0], self.x_lims_gen[0][1], num_points_x),
                torch.linspace(self.x_lims_gen[1][0], self.x_lims_gen[1][1], num_points_y),
            )
            z_0s = torch.vstack([xx.reshape(-1), yy.reshape(-1)]).T.to(output.z_generated.device)

            gen_output = model.solve(z_0s, model.drift_forward, model.diffusion_forward, model.time_values)
            gen_images = self.net.decode(gen_output[-1].cpu().detach()).cpu().detach().numpy()
            gen_output = gen_output.cpu().detach().numpy()
            for i in range(num_points_x * num_points_y):
                ab = AnnotationBbox(OffsetImage(gen_images[i].reshape((28, 28))),
                                    (gen_output[0, i, 0], gen_output[0, i, 1]),
                                    frameon=False)
                ax_generative.add_artist(ab)

            return [fig], ["results"]

        else:
            raise ValueError("Latent dims must be 2")


class Gaussian50D(BaseDataGenerator):
    def __init__(self, prior_dataset: BaseDataGenerator = None):
        super().__init__(prior_dataset)
        # Data properties
        self.n_train: int = 300
        self.n_test: int = 1000
        self.observed_dims: int = 50
        self.mean = 0.1

        self.x_lims = [[-10, 10], [-10, 10]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.prior_dataset is not None:
            if type(self.prior_dataset) is Gaussian50D:
                self.prior_dataset.mean = -self.mean
            self.prior_dataset.setup(stage)
        # Train
        self.xs_train = distributions.MultivariateNormal(loc=torch.tensor([0.]*50) + self.mean,
                                                         scale_tril=torch.eye(50)).sample((self.n_train,))
        # Test
        self.xs_test = distributions.MultivariateNormal(loc=torch.tensor([0.]*50) + self.mean,
                                                        scale_tril=torch.eye(50)).sample((self.n_test,))

    def plot_results(self, output: Output, model: Model, metrics: Metrics) \
            -> Tuple[List[Figure], List[str]]:
        z_values_forward = output.z_values_forward.cpu().detach().numpy()
        z_values_backward = output.z_values_backward.cpu().detach().numpy()
        z_generated = output.z_generated.cpu().detach().numpy()

        print(f"Mean prior: {np.mean(z_values_backward[:, -1])}")
        print(f"Mean data: {np.mean(z_generated[:, -1])}")
        print(f"Var prior: {np.mean(np.var(z_values_backward[:, -1], axis=-1))}")
        print(f"Var data: {np.mean(np.var(z_generated[:, -1], axis=-1))}")

        return [], []


datasets_dict = {
    "gaussian": Gaussian,
    # Experiments
    "toy_experiment_blobs_2d": Blobs2D,
    "toy_experiment_circle": Circle,
    "toy_experiment_circle_and_blob_concentric": CircleAndBlobConcentric,
    "toy_experiment_circle_and_blob": CircleAndBlob,
    "toy_experiment_banana": Banana,
    "toy_experiment_dual_moon": DualMoon,
    "toy_experiment_uneven_circle": UnevenCircle,
    "mnist_2d": MNIST2D,
    "gaussian_50d": Gaussian50D
}
