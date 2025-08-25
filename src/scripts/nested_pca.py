import argparse
import glob
import os
import typing
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from rignak.src.logging_utils import logger
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from rignak.src.lazy_property import LazyProperty
from src.modules.custom_objects import CUSTOM_OBJECTS
from src.output_spaces.custom.nested.nested_tags import Category, COLORS
from src.trainers.image_to_tag_trainers.run.nested_categorizer import DanbooruNestedCategorizerTrainer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dataset_pattern", type=str, required=False)
    parser.add_argument("--model_folder", type=str, required=True)
    return parser.parse_args()


@dataclass
class Processor:
    model_folder: str
    json_dataset_pattern: typing.Optional[str] = None

    n_frames: int = 20
    frame_shape: typing.Tuple[int, int] = (32, 32)

    _filenames: typing.Optional[typing.List[str]] = None
    _trainer: typing.Optional[DanbooruNestedCategorizerTrainer] = None
    _model: typing.Optional[tf.keras.models.Model] = None
    _outputs: typing.Optional[np.ndarray] = None
    _projected_outputs: typing.Optional[np.ndarray] = None
    _truths: typing.Optional[np.ndarray] = None

    @property
    def output_folder(self) -> str:
        return f"{self.model_folder}/manifolds"

    @LazyProperty
    def trainer(self) -> DanbooruNestedCategorizerTrainer:
        kwargs = {'on_start': False}
        if self.json_dataset_pattern:
            kwargs['pattern'] = self.json_dataset_pattern
        trainer = DanbooruNestedCategorizerTrainer(**kwargs)

        trainer.model_wrapper.model.load_weights(self.model_folder + '/model.h5')
        return trainer

    @LazyProperty
    def model(self) -> tf.keras.Model:
        logger("Extract intermediate model", indent=1)

        intermediate_layers = [
            self.trainer.model_wrapper.model.get_layer(name=f"dense_{i}").output
            for i in range(1, 2 * len(self.trainer.output_space.categories), 2)
        ]

        model = tf.keras.Model(
            inputs=self.trainer.model_wrapper.model.input,
            outputs=intermediate_layers
        )

        logger("Sanity check: all outputs should have a shape of (None, 16):", indent=1)
        for i, output in enumerate(model.output):
            if output.shape[1] != 16:
                logger(f"{i}: {output.shape}", level="warning")
        logger("Sanity check end", indent=-1)

        model.summary()

        logger("Load intermediate model OK", indent=-1)
        return model

    @LazyProperty
    def filenames(self) -> typing.List[str]:
        return list(self.trainer.output_space.samples.keys())

    @LazyProperty
    def outputs(self) -> np.ndarray:
        logger("Get intermediate_outputs", indent=1)

        n = int(len(self.filenames) / self.trainer.model_wrapper.batch_size)
        logger.set_iterator(n)

        outputs = np.zeros((len(self.filenames), len(self.model.outputs), self.model.outputs[0].shape[1]))
        truths = np.zeros((len(self.filenames), len(self.trainer.output_space)))
        for i in range(n):
            if i>10:
                break
            imin = i * self.trainer.model_wrapper.batch_size
            imax = imin + self.trainer.model_wrapper.batch_size
            filenames = self.filenames[imin:imax]

            inputs, truth = self.trainer.callback_generator.batch_processing(filenames)
            outputs[imin:imax] = np.array(self.model.predict(inputs, verbose=0)).swapaxes(0, 1)
            truths[imin:imax] = truth
            logger.iterate()

        outputs  =outputs[:imax]
        truths  = truths[:imax]

        self._truths = truths

        logger("Get intermediate_outputs OK", indent=-1)
        return outputs

    @LazyProperty
    def projected_outputs(self) -> typing.Dict[str, np.ndarray]:
        logger("Get projected_outputs", indent=1)
        projected_outputs = {}

        for i, category in enumerate(self.trainer.output_space.categories):
            pca = PCA(n_components=2)
            projected_output = pca.fit_transform(self.outputs[:, i])
            projected_outputs[category.name] = projected_output

        logger("Get projected_outputs OK", indent=-1)
        return projected_outputs

    def get_point_kwargs(self, category: Category, truth: np.ndarray):
        if not truth.sum():
            marker = '.'
            color = 'black'
            label = "Nothing"
        else:
            marker = "+"
            index = np.argmax(truth)
            label = category.labels[index]
            if '_' in label and label.split('_')[0] in COLORS:
                color = label.split('_')[0]
            else:
                color = COLORS[index % len(COLORS)]
            if color == "white":
                color = "gainsboro"
        kwargs = {'marker': marker, 'color': color, 'label': label}
        return kwargs

    def plot(self) -> None:
        logger("Plotting manifolds", indent=1)

        stride = int(len(self.filenames) // self.n_frames)

        imin = 0
        for i, (category_name, projected_output) in enumerate(self.projected_outputs.items()):
            category = self.trainer.output_space.categories[i]
            logger(f"Plot for {category_name}", indent=1)
            imax = imin + len(category)

            fig, ax = plt.subplots()
            logger.set_iterator(len(projected_output),percentage_threshold=1)
            for j, array in enumerate(projected_output):
                kwargs = self.get_point_kwargs(category, self._truths[j, imin:imax])
                ax.scatter(*array, **kwargs)
                logger.iterate()
            ax.legend()

            # Remove duplicated labels.
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            ax.grid()

            # for (x, y), filename in zip(projected_output[::stride], self.filenames[::stride]):
            #     img = Image.open(filename).resize(self.frame_shape)
            #     offset_img = OffsetImage(img, zoom=0.5)
            #     ann_box = AnnotationBbox(offset_img, (x, y), frameon=False)
            #     ax.add_artist(ann_box)


            imin = imax

            filename = f"{self.output_folder}/{category_name}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            logger(f"Saved manifold display to {filename}", indent=-1)

        logger("Plotting manifolds OK", indent=-1)

    def run(self) -> None:
        logger("Starting manifold creation.")
        self.plot()
        logger("Manifold creation finished.")


if __name__ == "__main__":
    # python src/scripts/nested_pca.py --model_folder "/ssd/OneDrive/Mes_documents/Documents/python_scripts/ML/.tmp/DanbooruNested_CategorizerWrapper_20250818_105641"

    args = get_args()
    Processor(model_folder=args.model_folder).run()
