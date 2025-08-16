import math
import typing

import numpy as np
import tensorflow as tf

from src.modules.blocks.convolution_block import ConvolutionBlock
from src.modules.blocks.deconvolution_block import DeconvolutionBlock


def build_encoder(
        current_layer: tf.keras.layers.Layer,
        layer_kernels: typing.Sequence[int],
        n_stride: int,
        superseeded_conv_layer: typing.Optional[tf.keras.layers.Layer] = None,
        superseeded_conv_kwargs: typing.Optional[dict] = None,
) -> typing.Tuple[tf.keras.layers.Layer, typing.Sequence[tf.keras.layers.Layer]]:
    inherited_layers = []
    for n_kernels in layer_kernels:
        current_layer, unpooled_layer = ConvolutionBlock(
            n_kernels,
            n_stride,
            superseeded_conv_layer=superseeded_conv_layer,
            superseeded_conv_kwargs=superseeded_conv_kwargs,
        )(current_layer)
        inherited_layers.append(unpooled_layer)
    return current_layer, inherited_layers


def build_decoder(
        current_layer: tf.keras.layers.Layer,
        inherited_layers: typing.Sequence[tf.keras.layers.Layer],
        layer_kernels: typing.Sequence[int],
        n_stride: int,
        superseeded_conv_layer: typing.Optional[tf.keras.layers.Layer] = None,
        superseeded_conv_kwargs: typing.Optional[dict] = None,
) -> tf.keras.layers.Layer:
    if not inherited_layers:
        inherited_layers = [None for _ in layer_kernels]
    for n_kernels, inherited_layer in zip(layer_kernels[::-1], inherited_layers[::-1]):
        current_layer = DeconvolutionBlock(
            n_kernels,
            n_stride,
            superseeded_conv_layer=superseeded_conv_layer,
            superseeded_conv_kwargs=superseeded_conv_kwargs
        )(current_layer, inherited_layer)
    return current_layer


def get_embedding(
        embedding_min_frequency: float,
        embedding_max_frequency: float,
        embedding_dims: int
) -> typing.Callable[[tf.Tensor], tf.Tensor]:
    min_frequency = np.log(embedding_min_frequency)
    max_frequency = np.log(embedding_max_frequency)
    frequencies = np.exp(np.linspace(min_frequency, max_frequency, embedding_dims // 2))
    angular_speeds = tf.keras.backend.cast(2.0 * math.pi * frequencies, "float32")

    def sinusoidal_embedding(x: tf.Tensor) -> tf.Tensor:
        embeddings = tf.keras.backend.concatenate(
            [tf.keras.backend.sin(angular_speeds * x), tf.keras.backend.cos(angular_speeds * x)], axis=3
        )
        return embeddings

    return sinusoidal_embedding
