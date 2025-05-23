import typing

import numpy as np
import tensorflow as tf
from Rignak.lazy_property import LazyProperty
from Rignak.logging_utils import logger

from src.callbacks.example_callback import ExampleCallback
from src.callbacks.example_callback_with_logs import ExampleCallbackWithLogs
from src.callbacks.plotters.image_to_tag.confuson_matrice_plotter import ConfusionMatricePlotter
from src.callbacks.plotters.image_to_tag.image_to_tag_example_plotter import ImageToTagExamplePlotter
from src.generators.base_generators import BatchGenerator
from src.generators.image_to_tag.classification_generator import ClassificationGenerator
from src.models.image_to_tag.categorizer_wrapper import CategorizerWrapper
from src.trainers.trainer import Trainer


class CategorizerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_generator: typing.Type = ClassificationGenerator

    @property
    def output_shape(self) -> typing.Sequence[int]:
        return self.original_training_generator.output_space.n,

    @property
    def get_model_wrapper(self) -> typing.Type:
        return CategorizerWrapper

    @LazyProperty
    def class_weights(self) -> np.ndarray:
        return self.original_training_generator.output_space.class_weights


    @LazyProperty
    def callback_generator(self, **kwargs) -> BatchGenerator:
        logger(f"Setup callback_generator", indent=1)
        generator = self.get_generator(self.validation_filenames, self.batch_size, self.input_shape, **kwargs)
        logger(f"Setup callback_generator OK", indent=-1)
        return generator

    @LazyProperty
    def callbacks(self) -> typing.Sequence[tf.keras.callbacks.Callback]:
        inputs, outputs = next(self.callback_generator)
        callbacks = [
            *super().callbacks,
            ExampleCallback(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/examples",
                function=ImageToTagExamplePlotter(
                    inputs,
                    outputs,
                    self.model_wrapper,
                    self.callback_generator.output_space
                )
            ),
            ExampleCallbackWithLogs(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/confusion",
                function=ConfusionMatricePlotter(
                    self.callback_generator,
                    self.validation_steps,
                    self.model_wrapper
                ),
                keep_all_epochs=False
            )
        ]
        return callbacks
