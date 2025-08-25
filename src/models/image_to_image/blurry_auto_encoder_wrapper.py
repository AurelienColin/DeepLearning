import typing

import tensorflow as tf

from rignak.src.lazy_property import LazyProperty
from src.losses.losses import fourth_channel_mae
from src.models.image_to_image.auto_encoder_wrapper import AutoEncoderWrapper


class BlurryAutoEncoderWrapper(AutoEncoderWrapper):

    @LazyProperty
    def output_shape(self) -> typing.Sequence[int]:
        return *self.input_shape[:-1], self.input_shape[-1] + 1

    @LazyProperty
    def metrics(self) -> typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        return *super().metrics, fourth_channel_mae
