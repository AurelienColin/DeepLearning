import os.path
import typing

import numpy as np
import tensorflow as tf
from rignak.lazy_property import LazyProperty
from rignak.logging_utils import logger

from output_spaces.custom.nested.nested_space import NestedSpace
from src.callbacks.example_callback import ExampleCallback
from src.callbacks.example_callback_with_logs import ExampleCallbackWithLogs
from callbacks.plotters.image_to_tag.confusion_matrix.nested_confuson_matrice_plotter import \
    NestedConfusionMatricePlotter
from callbacks.plotters.image_to_tag.example_plotter.nested_example_plotter import ImageToNestedTagExamplePlotter
from src.models.image_to_tag.nested_categorizer_wrapper import CategorizerWrapper
from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer
import glob
import json
from src.generators.base_generators import BatchGenerator, compose_generators
from src.generators.image_to_tag.classification_generator import ClassificationGenerator


class NestedCategorizerTrainer(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_filename = glob.glob(self.pattern)[0]
        self.output_space = NestedSpace(self.json_filename)

    @property
    def get_model_wrapper(self) -> typing.Type:
        return CategorizerWrapper


    def get_base_generator(self, *args, base_generator: typing.Optional[BatchGenerator] = None) -> BatchGenerator:
        generator = ClassificationGenerator(*args, output_space=self.output_space)
        return generator


    def get_generator(self, *args, base_generator: typing.Optional[BatchGenerator] = None) -> BatchGenerator:
        if base_generator is None:
            base_generator = self.base_generator(*args, **self.generator_kwargs, output_space=self.output_space)
        return compose_generators(base_generator, self.post_process_generator_classes)

    @LazyProperty
    def callbacks(self) -> typing.Sequence[tf.keras.callbacks.Callback]:
        inputs, outputs = next(self.callback_generator)
        callbacks = [
            *super(CategorizerTrainer, self).callbacks,
            ExampleCallback(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/examples",
                function=ImageToNestedTagExamplePlotter(
                    inputs,
                    outputs,
                    self.model_wrapper,
                    self.callback_generator.output_space
                )
            ),
            ExampleCallbackWithLogs(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/confusion",
                function=NestedConfusionMatricePlotter(
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

        with open(self.json_filename, 'r') as file:
            data = json.load(file)

        filenames = [entry['filename'] for entry in data]
        filenames = [filename for filename in filenames if os.path.exists(filename)]

        training_filenames = np.random.choice(filenames, size=int(0.75 * len(filenames)), replace=False)
        validation_filenames = [filename for filename in filenames if filename not in training_filenames]

        self._training_filenames = training_filenames
        self._validation_filenames = validation_filenames

        logger(f"Setup filenames OK", indent=-1)
