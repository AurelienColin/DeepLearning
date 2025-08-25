import typing

import tensorflow as tf
from rignak.src.lazy_property import LazyProperty

from src.callbacks.example_callback import ExampleCallback
from src.callbacks.plotters.image_to_image.image_to_image_example_plotter import ImageToImageExamplePlotter
from src.generators.image_to_image.autoencoder_generator import AutoEncoderGenerator
from src.models.image_to_image.auto_encoder_wrapper import AutoEncoderWrapper
from src.trainers.trainer import Trainer


class AutoEncoderTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_generator = AutoEncoderGenerator

    @property
    def get_model_wrapper(self) -> typing.Type:
        return AutoEncoderWrapper

    @LazyProperty
    def callbacks(self) -> typing.Sequence[tf.keras.callbacks.Callback]:
        callbacks = [
            *super().callbacks,
            ExampleCallback(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/.examples",
                function=ImageToImageExamplePlotter(*next(self.callback_generator), self.model_wrapper)
            ),
        ]
        return callbacks
