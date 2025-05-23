import typing

import tensorflow as tf

from src.modules.layers.atrous_conv2d import AtrousConv2D


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_kernels: int, n_stride: int, **kwargs):
        super().__init__(**kwargs)
        self.n_kernels: int = n_kernels
        self.n_stride: int = n_stride
        self.batch_norm: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()
        self.atrous_conv2ds: typing.Sequence[tf.keras.layers.Layer] = (
            AtrousConv2D(self.n_kernels, activation='swish', n_stride=self.n_stride),
            AtrousConv2D(self.n_kernels, activation=None, n_stride=self.n_stride)
        )
        self.add: tf.keras.layers.Layer = tf.keras.layers.Add()

        self.residual_conv: typing.Optional[tf.keras.layers.Layer] = None

    def build(self, input_shape: typing.Sequence[int]) -> None:
        if input_shape[-1] != self.n_kernels:
            self.residual_conv = tf.keras.layers.Conv2D(self.n_kernels, kernel_size=(1, 1))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        residual = self.residual_conv(inputs) if self.residual_conv else inputs

        x = self.batch_norm(inputs)
        for conv_layer in self.atrous_conv2ds:
            x = conv_layer(x)

        return self.add([x, residual])

    def get_config(self) -> typing.Dict:
        config = super().get_config()
        config.update(dict(n_kernels=self.n_kernels, n_stride=self.n_stride))
        return config

    @classmethod
    def from_config(cls, config: typing.Dict):
        return cls(**config)
