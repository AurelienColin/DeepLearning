from dataclasses import dataclass
import tensorflow as tf
from rignak.src.logging_utils import logger
from rignak.src.lazy_property import LazyProperty
from src.trainers.image_to_tag_trainers.run.nested_categorizer import DanbooruNestedCategorizerTrainer
import typing
import numpy as np


@dataclass
class Processor:
    model_folder: str
    json_dataset_pattern: typing.Optional[str] = None

    _filenames: typing.Optional[typing.List[str]] = None
    _trainer: typing.Optional[DanbooruNestedCategorizerTrainer] = None
    _model: typing.Optional[tf.keras.models.Model] = None
    _outputs: typing.Optional[np.ndarray] = None
    _projected_outputs: typing.Optional[np.ndarray] = None
    _truths: typing.Optional[np.ndarray] = None

    @LazyProperty
    def trainer(self) -> DanbooruNestedCategorizerTrainer:
        """Loads the trainer and model weights from the specified folder."""
        logger("Loading trainer and weights", indent=1)
        kwargs = {'on_start': False}
        # This trainer setup seems to require a dataset pattern,
        # even if just for initialization. We handle this in the child class.
        if hasattr(self, 'json_dataset_pattern') and self.json_dataset_pattern:
            kwargs['pattern'] = self.json_dataset_pattern

        trainer = DanbooruNestedCategorizerTrainer(**kwargs)
        trainer.model_wrapper.model.load_weights(f"{self.model_folder}/model.h5")
        logger("Trainer and weights loaded OK", indent=-1)
        return trainer

    @LazyProperty
    def model(self) -> tf.keras.Model:
        """Builds an intermediate model to extract hidden layer outputs."""
        logger("Extracting intermediate model", indent=1)

        # Extracts the output of dense layers, which likely represent embeddings for each category
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
                logger(f"Layer {i} has unexpected shape: {output.shape}", level="warning")
        logger("Sanity check end", indent=-1)

        model.summary()
        logger("Intermediate model built OK", indent=-1)
        return model
