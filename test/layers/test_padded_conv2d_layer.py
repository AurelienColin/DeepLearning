import typing
import pytest
from ML.src.modules.layers.padded_conv2d import PaddedConv2D
import tensorflow as tf

BATCH_SIZE: int = 4
INPUT_SHAPE: typing.Tuple[int, int] = (16, 24)
N_KERNELS: int = 15
N_INPUT_CHANNELS: int = 2



@pytest.fixture(scope='module')
def model_without_dilation() -> tf.keras.models.Model:
    input_layer = tf.keras.layers.Input(shape=(*INPUT_SHAPE, N_INPUT_CHANNELS))
    layer = PaddedConv2D(n_kernels=N_KERNELS, dilation_rate=1, activation=None, name='layer')(input_layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=layer, name="model_without_dilation")
    model.summary()
    return model


def test_number_of_trainable_weights_without_dilation(model_without_dilation: tf.keras.models.Model) -> None:
    kernel = model_without_dilation.get_layer('layer').get_weights()
    assert kernel[0].shape == (3, 3, N_INPUT_CHANNELS, N_KERNELS)
    assert kernel[1].shape == (N_KERNELS, )


def test_output_shape_without_dilation(model_without_dilation: tf.keras.models.Model) -> None:
    assert model_without_dilation.output_shape == (None, *INPUT_SHAPE, N_KERNELS)


@pytest.fixture(scope='module')
def model_with_dilation() -> tf.keras.models.Model:
    input_layer = tf.keras.layers.Input(shape=(*INPUT_SHAPE, N_INPUT_CHANNELS))
    layer = PaddedConv2D(n_kernels=N_KERNELS, dilation_rate=3, activation=None, name='layer')(input_layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=layer, name="model_with_dilation")
    model.summary()
    return model


def test_number_of_trainable_weights_with_dilation(model_with_dilation: tf.keras.models.Model) -> None:
    kernel = model_with_dilation.get_layer('layer').get_weights()
    assert kernel[0].shape == (3, 3, N_INPUT_CHANNELS, N_KERNELS)
    assert kernel[1].shape == (N_KERNELS, )


def test_output_shape_with_dilation(model_with_dilation: tf.keras.models.Model) -> None:
    assert model_with_dilation.output_shape == (None, *INPUT_SHAPE, N_KERNELS)
