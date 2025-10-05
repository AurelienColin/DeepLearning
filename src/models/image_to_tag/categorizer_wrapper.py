import typing
from dataclasses import dataclass

import tensorflow as tf
from rignak.src.lazy_property import LazyProperty

from src.losses import losses
from src.models.model_wrapper import ModelWrapper
from src.modules.module import build_encoder
from src.config import DEFAULT_ACTIVATION

@dataclass
class CategorizerWrapper(ModelWrapper):
    layer_kernels: typing.Sequence[int] = (32, 64, 128)

    _encoded_layer: typing.Optional[tf.keras.layers.Layer] = None
    _encoded_inherited_layers: typing.Optional[typing.Sequence[tf.keras.layers.Layer]] = None


    @LazyProperty
    def loss(self) -> typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        return losses.Loss((losses.cross_entropy,), class_weights=self.training_generator.output_space.class_weights)

    @LazyProperty
    def metrics(self) -> typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        return losses.one_minus_dice, losses.std_difference

    @LazyProperty
    def output_layer(self) -> tf.keras.layers.Layer:
        current_layer, _ = build_encoder(
            self.input_layer,
            self.layer_kernels,
            superseeded_conv_layer=self.superseeded_conv_layer,
            superseeded_conv_kwargs=self.superseeded_conv_kwargs,
        )
        current_layer = tf.keras.layers.GlobalAveragePooling2D()(current_layer)

        n_intermediate = abs(int((self.layer_kernels[-1] - self.output_shape[-1]) / 2))
        current_layer = tf.keras.layers.Dense(n_intermediate, activation=DEFAULT_ACTIVATION, )(current_layer)
        output_layer = tf.keras.layers.Dense(self.output_shape[-1], activation="sigmoid", )(current_layer)
        return output_layer
