import os
import typing
from dataclasses import dataclass
import numpy as np
from rignak.src.logging_utils import logger
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from rignak.src.lazy_property import LazyProperty

from src.scripts.utils.processors.processor import  Processor

@dataclass
class FileProcessor(Processor):
    """
    Processes a given list of image files to visualize where they fall on the
    model's learned manifolds. It plots the actual images on the 2D projection.
    """
    filenames: typing.Optional[typing.List[str]] = None
    frame_shape: typing.Tuple[int, int] = (32, 32)


    @property
    def output_folder(self) -> str:
        """Defines the directory to save the output plots."""
        return f"{self.model_folder}/manifolds_from_files"

    @LazyProperty
    def outputs(self) -> np.ndarray:
        """Runs model predictions to get intermediate layer outputs for the given files."""
        logger(f"Getting intermediate outputs from {len(self.filenames)} files", indent=1)

        # The trainer and its data loader are still needed for image preprocessing
        generator = self.trainer.callback_generator
        batch_size = self.trainer.model_wrapper.batch_size
        n_batches = int(np.ceil(len(self.filenames) / batch_size))
        logger.set_iterator(n_batches)

        outputs_list = []
        for i in range(n_batches):
            imin = i * batch_size
            imax = imin + batch_size
            batch_filenames = self.filenames[imin:imax]

            if not batch_filenames:
                continue

            # We only need the preprocessed inputs, not the truths
            inputs, _ = generator.batch_processing(batch_filenames)
            prediction = self.model.predict(inputs, verbose=0)
            outputs_list.append(np.array(prediction).swapaxes(0, 1))
            logger.iterate()

        outputs = np.vstack(outputs_list)
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

    def plot(self) -> None:
        """Generates and saves a 2D plot for each manifold with images overlaid."""
        logger("Plotting manifolds with images", indent=1)
        os.makedirs(self.output_folder, exist_ok=True)

        for i, (category_name, projected_output) in enumerate(self.projected_outputs.items()):
            logger(f"Plotting for category: {category_name}", indent=1)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(projected_output[:, 0], projected_output[:, 1], c='gray', alpha=0.3)
            ax.grid(True)
            ax.set_title(f"PCA Manifold for '{category_name}' with Image Samples")

            # Overlay images on the plot
            for (x, y), filename in zip(projected_output, self.filenames):
                try:
                    img = Image.open(filename).resize(self.frame_shape)
                    offset_img = OffsetImage(img, zoom=1.0, alpha=0.9)
                    ann_box = AnnotationBbox(offset_img, (x, y), frameon=False)
                    ax.add_artist(ann_box)
                except Exception as e:
                    logger.warning(f"Could not process image {filename}: {e}")

            # Save the figure
            filename = f"{self.output_folder}/{category_name}.png"
            plt.savefig(filename)
            plt.close(fig)
            logger(f"Saved manifold display to {filename}", indent=-1)

        logger("Plotting manifolds OK", indent=-1)

    def run(self) -> None:
        """Main execution method."""
        logger("Starting manifold creation from file list.")
        if not self.filenames:
            logger.error("No filenames provided for FileProcessor.")
            return
        self.plot()
        logger("Manifold creation finished.")