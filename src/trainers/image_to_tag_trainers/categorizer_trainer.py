import os.path
import typing

import numpy as np
import tensorflow as tf
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger

from src.callbacks.example_callback import ExampleCallback
from src.callbacks.example_callback_with_logs import ExampleCallbackWithLogs
from src.callbacks.plotters.image_to_tag.confusion_matrix.flat_confusion_matrix_plotter import FlatConfusionMatricePlotter
from src.callbacks.plotters.image_to_tag.example_plotter.flat_example_plotter import ImageToFlatTagExamplePlotter
from src.generators.base_generators import BatchGenerator
from src.generators.image_to_tag.classification_generator import ClassificationGenerator
from src.models.image_to_tag.categorizer_wrapper import CategorizerWrapper
from src.trainers.trainer import Trainer
import glob


class CategorizerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_generator: typing.Type = ClassificationGenerator

    @property
    def output_shape(self) -> typing.Sequence[int]:
        return len(self.original_training_generator.output_space),

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
                function=ImageToFlatTagExamplePlotter(
                    inputs,
                    outputs,
                    self.model_wrapper,
                    self.callback_generator.output_space
                )
            ),
            ExampleCallbackWithLogs(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/confusion",
                function=FlatConfusionMatricePlotter(
                    self.callback_generator,
                    self.validation_steps,
                    self.model_wrapper
                ),
                keep_all_epochs=False
            )
        ]
        return callbacks

    def set_filenames(self) -> None:
        logger(f"Setup filenames", indent=1)
        root, base_pattern = os.path.split(self.pattern)
        class_names = glob.glob(root)

        training_filenames = []
        validation_filenames = []
        for class_name in class_names:
            class_pattern = os.path.join(class_name, base_pattern)
            class_training_filenames, class_validation_filenames = self.split_names_on_pattern(class_pattern)
            training_filenames.extend(class_training_filenames)
            validation_filenames.extend(class_validation_filenames)

        self._training_filenames = training_filenames
        self._validation_filenames = validation_filenames

        logger(f"Setup filenames OK", indent=-1)
