import typing
from dataclasses import dataclass

import tensorflow as tf
from rignak.lazy_property import LazyProperty

from src.losses import losses
from src.models.model_wrapper import ModelWrapper
from src.modules.module import build_encoder


@dataclass
class CategorizerWrapper(ModelWrapper):
    layer_kernels: typing.Sequence[int] = (32, 48, 64, 96, 128)

    _encoded_layer: typing.Optional[tf.keras.layers.Layer] = None
    _encoded_inherited_layers: typing.Optional[typing.Sequence[tf.keras.layers.Layer]] = None

    superseeded_conv_layer: typing.Optional[tf.keras.layers.Layer] = None
    superseeded_conv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None

    @LazyProperty
    def loss(self) -> typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        return losses.Loss(
            (losses.cross_entropy_positive,),
            class_weights=self.training_generator.output_space.class_weights
        )

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
        current_layer = tf.keras.layers.Dense(self.layer_kernels[-1], activation="swish")(current_layer)

        output_layers = []
        for category in self.training_generator.output_space.categories:
            intermediate_layer = tf.keras.layers.Dense(16, activation="swish")(current_layer)
            output_layer = tf.keras.layers.Dense(len(category), activation="softmax")(intermediate_layer)
            output_layers.append(output_layer)

        output_layer = tf.keras.layers.concatenate(output_layers)
        return output_layer
