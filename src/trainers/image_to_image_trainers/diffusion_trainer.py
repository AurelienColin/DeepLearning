import typing

import tensorflow as tf
from rignak.lazy_property import LazyProperty

from src.callbacks.example_callback import ExampleCallback
from src.callbacks.plotters.image_to_image.diffusion_example_plotter import DiffusionExamplePlotter
from src.callbacks.plotters.image_to_image.diffusion_random_plotter import DiffusionRandomPlotter
from src.models.image_to_image.diffusion_model_wrapper import DiffusionModelWrapper
from src.trainers.image_to_image_trainers.autoencoder_trainer import AutoEncoderTrainer


class DiffusionTrainer(AutoEncoderTrainer):

    @property
    def get_model_wrapper(self) -> typing.Type:
        return DiffusionModelWrapper

    @LazyProperty
    def callbacks(self) -> typing.Sequence[tf.keras.callbacks.Callback]:
        inputs, outputs = next(self.callback_generator)

        callbacks = [
            *super(AutoEncoderTrainer, self).callbacks,
            ExampleCallback(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/.examples",
                function=DiffusionRandomPlotter(inputs, outputs, self.model_wrapper),
            ),
            ExampleCallback(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/.generate",
                function=DiffusionExamplePlotter(inputs, outputs, self.model_wrapper)
            )
        ]
        return callbacks
