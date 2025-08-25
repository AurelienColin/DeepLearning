import typing

import numpy as np
import tensorflow as tf


class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, means: np.ndarray, stds: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def get_config(self) -> typing.Dict:
        config = super().get_config()
        config.update({'means': self.means.tolist(), 'stds': self.stds.tolist(), })
        return config

    @classmethod
    def from_config(cls, config: typing.Dict):
        means = np.array(config.pop('means'), dtype=np.float32)
        stds = np.array(config.pop('stds'), dtype=np.float32)
        return cls(means=means, stds=stds, **config)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return (inputs - self.means) / self.stds
