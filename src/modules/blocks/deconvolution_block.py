import typing

import tensorflow as tf

from src.modules.blocks.residual_block import ResidualBlock


class DeconvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, n_kernels: int, n_stride: int, **kwargs):
        super().__init__(**kwargs)
        self.n_kernels: int = n_kernels
        self.n_stride: int = n_stride
        self.upsampling: tf.keras.layers.Layer = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.concat: tf.keras.layers.Layer = tf.keras.layers.Concatenate()
        self.residual_block: tf.keras.layers.Layer = ResidualBlock(self.n_kernels, self.n_stride)

    def call(
            self,
            current_layer: tf.Tensor,
            inherited_layer: typing.Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        current_layer = self.upsampling(current_layer)
        if inherited_layer is not None:
            current_layer = self.concat([current_layer, inherited_layer])
        return self.residual_block(current_layer)

    def get_config(self) -> typing.Dict:
        config = super().get_config()
        config.update(dict(n_kernels=self.n_kernels, n_stride=self.n_stride))
        return config

    @classmethod
    def from_config(cls, config: typing.Dict):
        return cls(**config)
