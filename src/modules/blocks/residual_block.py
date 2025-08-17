import typing

import tensorflow as tf

from src.modules.layers.padded_conv2d import PaddedConv2D

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            n_kernels: int,
            superseeded_conv_layer: typing.Optional[tf.keras.layers.Layer] = None,
            superseeded_conv_kwargs: typing.Optional[dict] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_kernels: int = n_kernels
        self.batch_norm: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()

        self.superseeded_conv_layer: typing.Optional[tf.keras.layers.Layer] = superseeded_conv_layer
        self.superseeded_conv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = superseeded_conv_kwargs

        self.conv2ds: typing.Sequence[tf.keras.layers.Layer] = (
            self.get_convolution_layer(n_kernels=self.n_kernels, activation='swish', dilation_rate=3),
            self.get_convolution_layer(n_kernels=self.n_kernels, activation=None, dilation_rate=3)
        )
        self.add: tf.keras.layers.Layer = tf.keras.layers.Add()
        self.residual_conv: typing.Optional[tf.keras.layers.Layer] = None

    def get_convolution_layer(self, **kwargs) -> tf.keras.layers.Layer:
        if self.superseeded_conv_kwargs is not None:
            kwargs.update(self.superseeded_conv_kwargs)

        conv_layer_class = PaddedConv2D if self.superseeded_conv_layer is None else self.superseeded_conv_layer
        conv_layer = conv_layer_class(**kwargs)
        return conv_layer

    def build(self, input_shape: typing.Sequence[int]) -> None:
        if input_shape[-1] != self.n_kernels:
            self.residual_conv = tf.keras.layers.Conv2D(self.n_kernels, kernel_size=(1, 1))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        residual = self.residual_conv(inputs) if self.residual_conv else inputs

        x = self.batch_norm(inputs)
        for conv_layer in self.conv2ds:
            x = conv_layer(x)

        return self.add([x, residual])

    def get_config(self) -> typing.Dict:
        config = super().get_config()
        config.update(dict(n_kernels=self.n_kernels))
        return config

    @classmethod
    def from_config(cls, config: typing.Dict):
        return cls(**config)
