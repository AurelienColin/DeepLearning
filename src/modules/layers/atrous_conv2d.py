import typing

import numpy as np
import tensorflow as tf

from src.modules.layers.padded_conv2d import PaddedConv2D


class AtrousConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_kernels: int,
            n_stride: int,
            dilation_rate: int=1,
            activation: typing.Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_stride: int = n_stride
        self.n_kernels: int = n_kernels
        self.conv_layers: typing.Optional[typing.Sequence[tf.keras.layers.Layer]] = None
        self.activation: typing.Optional[str] = activation
        self.dilation_rate: int = dilation_rate

    def build(self, input_shape: typing.Sequence[int]) -> None:
        if self.n_stride != 1:
            kernel_counts = np.linspace(0, self.n_kernels, self.n_stride+1).astype(int)
            kernel_counts = np.ediff1d(kernel_counts)
        else:
            kernel_counts = [self.n_kernels]

        self.conv_layers = [
            PaddedConv2D(
                activation=self.activation,
                n_kernels=i_kernel,
                dilation_rate=dilation_rate  * self.dilation_rate,
                name=f"padded_conv_dilation_{dilation_rate}"
            )
            for dilation_rate, i_kernel in enumerate(kernel_counts, start=1)
        ]
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = [conv_layer(inputs) for conv_layer in self.conv_layers]
        return tf.keras.layers.Concatenate()(outputs)

    def get_config(self) -> typing.Dict:
        config = super().get_config()
        config.update(dict(
            n_kernels=self.n_kernels,
            activation=self.activation,
            n_stride=self.n_stride
        ))
        return config

    @classmethod
    def from_config(cls, config: typing.Dict):
        return cls(**config)
