import typing
from dataclasses import dataclass

import tensorflow as tf
import tensorflow.keras.backend as K
from rignak.src.lazy_property import LazyProperty

from src.losses.losses import edge_loss, mae, Loss
from src.models.model_wrapper import ModelWrapper
from src.modules.module import build_encoder, build_decoder


@dataclass
class AutoEncoderWrapper(ModelWrapper):
    layer_kernels: typing.Sequence[int] = (32, 64, 128)
    _output_shape: typing.Optional[typing.Sequence[int]] = None

    _encoded_layer: typing.Optional[tf.keras.layers.Layer] = None
    _encoded_inherited_layers: typing.Optional[typing.Sequence[tf.keras.layers.Layer]] = None


    superseeded_conv_layer: typing.Optional[tf.keras.layers.Layer] = None
    superseeded_conv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None

    def desactivate_edge_loss(self) -> None:
        self._loss = mae

    @LazyProperty
    def loss(self) -> typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        if self.output_shape[-1] in (3, 4):
            return Loss((edge_loss, mae), loss_weights=(0.1, 1))
        else:
            return mae

    @LazyProperty
    def metrics(self) -> typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        if self.output_shape[-1] in (3, 4):
            return edge_loss, mae
        else:
            return mae,

    def set_encoded_layers(self) -> None:
        current_layer, self._encoded_inherited_layers = build_encoder(
            self.input_layer,
            self.layer_kernels,
            superseeded_conv_layer=self.superseeded_conv_layer,
            superseeded_conv_kwargs=self.superseeded_conv_kwargs,
        )
        self._encoded_layer = tf.keras.layers.Lambda(lambda x: K.tanh(x))(current_layer)

    @LazyProperty
    def encoded_layer(self) -> tf.keras.layers.Layer:
        self.set_encoded_layers()
        return self._encoded_layer

    @property
    def encoded_inherited_layers(self) -> typing.Sequence[tf.keras.layers.Layer]:
        return []

    @LazyProperty
    def output_layer(self) -> tf.keras.layers.Layer:
        current_layer = build_decoder(
            self.encoded_layer,
            self.encoded_inherited_layers,
            self.layer_kernels,
            superseeded_conv_layer=self.superseeded_conv_layer,
            superseeded_conv_kwargs=self.superseeded_conv_kwargs,
        )
        output_layer = tf.keras.layers.Conv2D(
            self.output_shape[-1],
            activation="sigmoid",
            kernel_size=1
        )(current_layer)
        return output_layer
