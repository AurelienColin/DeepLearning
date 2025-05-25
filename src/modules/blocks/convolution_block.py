import typing

import tensorflow as tf

from src.modules.blocks.residual_block import ResidualBlock


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, n_kernels: int, n_stride: int, **kwargs):
        super().__init__(**kwargs)
        self.n_kernels: int = n_kernels
        self.n_stride: int = n_stride
        self.residual_block: tf.keras.layers.Layer = ResidualBlock(self.n_kernels, self.n_stride)
        self.pooling: tf.keras.layers.Layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

    def call(self, inputs: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        layer = self.residual_block(inputs)
        pooled_layer = self.pooling(layer)
        return pooled_layer, layer

    def get_config(self) -> typing.Dict:
        config = super().get_config()
        config.update(dict(n_kernels=self.n_kernels, n_stride=self.n_stride))
        return config

    @classmethod
    def from_config(cls, config: typing.Dict):
        return cls(**config)
