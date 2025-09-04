import os
import typing
from dataclasses import dataclass
import numpy as np
from rignak.src.logging_utils import logger
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from rignak.src.lazy_property import LazyProperty
from src.output_spaces.custom.nested.nested_tags import Category, COLORS

from src.scripts.utils.processors.processor import  Processor


@dataclass
class VideoProcessor(Processor):
    """
    Processes data from the original JSON dataset to visualize the model's learned manifolds.
    It plots data points and colors them according to their ground truth labels.
    """
    n_frames: int = 20
    frame_shape: typing.Tuple[int, int] = (32, 32)

    _truths: typing.Optional[np.ndarray] = None

    @property
    def output_folder(self) -> str:
        """Defines the directory to save the output plots."""
        return f"{self.model_folder}/manifolds_from_dataset"

    @LazyProperty
    def filenames(self) -> typing.List[str]:
        """Retrieves filenames from the trainer's dataset samples."""
        return list(self.trainer.output_space.samples.keys())

    @LazyProperty
    def outputs(self) -> np.ndarray:
        """Runs model predictions to get intermediate layer outputs and truths."""
        logger("Getting intermediate outputs from dataset", indent=1)
        batch_size = self.trainer.model_wrapper.batch_size
        n_batches = int(len(self.filenames) / batch_size)
        logger.set_iterator(n_batches)

        # Pre-allocate arrays for efficiency
        outputs = np.zeros((len(self.filenames), len(self.model.outputs), self.model.outputs[0].shape[1]))
        truths = np.zeros((len(self.filenames), len(self.trainer.output_space)))

        imax = 0
        for i in range(n_batches):
            imin = i * batch_size
            imax = imin + batch_size
            batch_filenames = self.filenames[imin:imax]

            inputs, truth = self.trainer.callback_generator.batch_processing(batch_filenames)
            prediction = self.model.predict(inputs, verbose=0)

            outputs[imin:imax] = np.array(prediction).swapaxes(0, 1)
            truths[imin:imax] = truth
            logger.iterate()

        # Trim arrays to the actual number of processed items
        outputs = outputs[:imax]
        self._truths = truths[:imax]

        logger("Finished getting intermediate outputs", indent=-1)
        return outputs

    @LazyProperty
    def projected_outputs(self) -> typing.Dict[str, np.ndarray]:
        """Projects high-dimensional outputs to 2D using PCA for visualization."""
        logger("Projecting outputs with PCA", indent=1)
        projected_outputs = {}
        for i, category in enumerate(self.trainer.output_space.categories):
            pca = PCA(n_components=2)
            projected_output = pca.fit_transform(self.outputs[:, i])
            projected_outputs[category.name] = projected_output
        logger("PCA projection finished", indent=-1)
        return projected_outputs

    def get_point_kwargs(self, category: Category, truth: np.ndarray) -> dict:
        """Determines plot style (color, marker) for a point based on its true label."""
        if not truth.sum():
            return {'marker': '.', 'color': 'black', 'label': "Nothing"}

        marker = "+"
        index = np.argmax(truth)
        label = category.labels[index]

        # Determine color from label name or default palette
        if '_' in label and label.split('_')[0] in COLORS:
            color = label.split('_')[0]
        else:
            color = COLORS[index % len(COLORS)]

        if color == "white":
            color = "gainsboro"  # Use a visible color instead of white

        return {'marker': marker, 'color': color, 'label': label}

    def plot(self) -> None:
        """Generates and saves a 2D scatter plot for each category manifold."""
        logger("Plotting manifolds", indent=1)
        os.makedirs(self.output_folder, exist_ok=True)

        imin = 0
        for i, (category_name, projected_output) in enumerate(self.projected_outputs.items()):
            category = self.trainer.output_space.categories[i]
            imax = imin + len(category)
            logger(f"Plotting for category: {category_name}", indent=1)

            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot each point with its corresponding style
            for j, point in enumerate(projected_output):
                kwargs = self.get_point_kwargs(category, self._truths[j, imin:imax])
                ax.scatter(*point, **kwargs)

            # Create a clean legend with unique labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.grid(True)
            ax.set_title(f"PCA Manifold for '{category_name}'")

            # Save the figure
            filename = f"{self.output_folder}/{category_name}.png"
            plt.savefig(filename)
            plt.close(fig)  # Close the figure to free up memory
            logger(f"Saved manifold display to {filename}", indent=-1)

            imin = imax

        logger("Plotting manifolds OK", indent=-1)
