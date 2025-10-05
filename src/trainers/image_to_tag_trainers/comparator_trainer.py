import os.path
import typing

import numpy as np
import tensorflow as tf
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger
from src.callbacks.example_callback import ExampleCallback
from src.callbacks.example_callback_with_logs import ExampleCallbackWithLogs
from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer
import glob
from src.generators.base_generators import BatchGenerator, compose_generators
from src.output_spaces.comparator_from_filesystem import ComparatorSpace
from src.models.image_to_tag.comparator_wrapper import Comparator
from src.generators.image_to_tag.comparator_generator import ComparatorGenerator
from src.callbacks.plotters.image_to_tag.example_plotter.comparator_example_plotter import ComparatorExamplePlotter
from src.callbacks.plotters.image_to_tag.confusion_matrix.comparator_matrice_plotter import \
    ComparatorConfusionMatricePlotter


class ComparatorTrainer(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_generator: typing.Type = ComparatorGenerator

        sources = glob.glob(self.pattern)
        self.output_space = ComparatorSpace(sources)

    @property
    def output_shape(self) -> typing.Sequence[int]:
        return self.original_training_generator.output_space.level

    @property
    def get_model_wrapper(self) -> typing.Type:
        return Comparator

    def get_base_generator(self, *args, base_generator: typing.Optional[BatchGenerator] = None) -> BatchGenerator:
        generator = ComparatorGenerator(*args, output_space=self.output_space)
        return generator

    def get_generator(self, *args, base_generator: typing.Optional[BatchGenerator] = None) -> BatchGenerator:
        if base_generator is None:
            base_generator = self.base_generator(*args, **self.generator_kwargs, output_space=self.output_space)
        return compose_generators(base_generator, self.post_process_generator_classes)

    @LazyProperty
    def post_process_generator_classes(self) -> typing.Sequence[typing.Type]:
        return ()

    @LazyProperty
    def callbacks(self) -> typing.Sequence[tf.keras.callbacks.Callback]:
        inputs, outputs = next(self.callback_generator)
        callbacks = [
            *super(CategorizerTrainer, self).callbacks,
            ExampleCallback(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/examples",
                function=ComparatorExamplePlotter(
                    inputs,
                    outputs,
                    self.model_wrapper,
                    self.callback_generator.output_space
                )
            ),
            ExampleCallbackWithLogs(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/confusion",
                function=ComparatorConfusionMatricePlotter(
                    self.callback_generator,
                    self.validation_steps,
                    self.model_wrapper
                ),
                keep_all_epochs=False
            )
        ]
        return callbacks

    def set_filenames(self) -> None:
        filenames = np.array(list(self.output_space.data.keys()))
        indices = np.arange(len(filenames))
        np.random.shuffle(indices)
        split_index = int(0.75 * len(filenames))

        training_filenames = filenames[indices[:split_index]]
        validation_filenames = filenames[indices[split_index:]]

        self._training_filenames = training_filenames
        self._validation_filenames = validation_filenames

        logger(f"Setup filenames OK", indent=-1)

    @staticmethod
    def generator_to_dataset(generator: BatchGenerator) -> tf.data.Dataset:
        sample_x, sample_y_list = next(generator)

        output_shapes = (
            tf.TensorShape((None, *sample_x.shape[1:])),
            tuple(tf.TensorShape((None, *y.shape[1:])) for y in sample_y_list)
        )
        output_types = (tf.float32, tuple(tf.float32 for _ in sample_y_list))

        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=output_shapes[0], dtype=output_types[0]),
                tuple(tf.TensorSpec(shape=s, dtype=t) for s, t in zip(output_shapes[1], output_types[1]))
            )
        )
        return dataset
