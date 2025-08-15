import typing
import pytest
from ML.src.modules.layers.atrous_conv2d import AtrousConv2D
import tensorflow as tf

BATCH_SIZE: int = 4
INPUT_SHAPE: typing.Tuple[int, int] = (16, 24)
N_KERNELS: int = 15
N_INPUT_CHANNELS: int = 2
N_STRIDE: int = 3

@pytest.fixture(scope='module')
def model() -> tf.keras.models.Model:
    input_layer = tf.keras.layers.Input(shape=(*INPUT_SHAPE, N_INPUT_CHANNELS))
    layer = AtrousConv2D(n_kernels=N_KERNELS, n_stride=N_STRIDE, activation=None, name='layer')(input_layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=layer, name="model")
    model.summary()
    return model


def test_number_of_trainable_weights(model: tf.keras.models.Model) -> None:
    kernel = model.get_layer('layer').get_weights()
    kernel_per_stride = int(N_KERNELS // N_STRIDE)
    assert kernel[0].shape == (3, 3, N_INPUT_CHANNELS, kernel_per_stride)
    assert kernel[1].shape == (kernel_per_stride,)
    assert kernel[2].shape == (3, 3, N_INPUT_CHANNELS, kernel_per_stride)
    assert kernel[3].shape == (kernel_per_stride,)
    assert kernel[4].shape == (3, 3, N_INPUT_CHANNELS, kernel_per_stride)
    assert kernel[5].shape == (kernel_per_stride,)


def test_output_shape(model: tf.keras.models.Model) -> None:
    assert model.output_shape == (None, *INPUT_SHAPE, N_KERNELS)
