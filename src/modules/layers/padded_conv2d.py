import typing

import tensorflow as tf
from rignak.src.logging_utils import logger


class PaddedConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_kernels: int,
            dilation_rate: int=1,
            activation: typing.Optional[str]=None,
            n_stride: int = 1,
            **kwargs):
        super().__init__(**kwargs)
        self.n_kernels: int = n_kernels
        self.dilation_rate: int = dilation_rate
        self.activation: typing.Optional[str] = activation
        self.conv_layer = tf.keras.layers.Conv2D(
            n_kernels,
            (3, 3),
            dilation_rate=dilation_rate,
            kernel_constraint=tf.keras.constraints.max_norm(2.),
            padding='valid',
            activation=self.activation
        )
        self.pad: typing.Tuple[int, int] = (dilation_rate, dilation_rate)


        if n_stride != 1:
            logger.warning(f"n_stride != 1 is not supported for PaddedConv2D "
                           f"but provided value is {n_stride}")
        self.n_stride: int = n_stride

    def get_config(self) -> typing.Dict:
        config = super().get_config()
        config.update({
            'n_kernels': self.n_kernels,
            'dilation_rate': self.dilation_rate,
            'activation': self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config: typing.Dict):
        return cls(**config)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv_layer(inputs)
        padding = ((0, 0), self.pad, self.pad, (0, 0))
        return tf.pad(x, padding, mode="REFLECT")
