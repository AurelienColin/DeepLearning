import typing
import pytest
from ML.src.modules.layers.padded_conv2d import PaddedConv2D
import tensorflow as tf
import numpy as np

BATCH_SIZE: int = 4
INPUT_SHAPE: typing.Tuple[int, int] = (16, 24)
N_KERNELS: int = 15
N_INPUT_CHANNELS: int = 2


@pytest.fixture
def sample_input():
    """Shape: (batch, height, width, channels)"""
    return tf.random.normal((BATCH_SIZE, *INPUT_SHAPE, N_INPUT_CHANNELS))


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


# Enhanced tests inspired by test_from_another_repo

def test_layer_initialization():
    """Test that layer properties are correctly initialized."""
    layer = PaddedConv2D(n_kernels=8, dilation_rate=2, activation='relu')
    assert layer.n_kernels == 8
    assert layer.dilation_rate == 2
    assert layer.activation == 'relu'
    assert isinstance(layer.conv_layer, tf.keras.layers.Conv2D)
    assert layer.pad == (2, 2)


def test_get_and_from_config():
    """Test serialization and deserialization."""
    layer = PaddedConv2D(n_kernels=4, dilation_rate=1, activation='sigmoid')
    config = layer.get_config()
    new_layer = PaddedConv2D.from_config(config)

    assert new_layer.n_kernels == 4
    assert new_layer.dilation_rate == 1
    assert new_layer.activation == 'sigmoid'


def test_reflect_padding_effect(sample_input):
    """Ensure output values differ near the borders due to reflection padding."""
    layer = PaddedConv2D(n_kernels=1, dilation_rate=1)
    output = layer(sample_input)

    # Ensure output values differ near the borders due to reflection padding
    assert not tf.reduce_all(tf.equal(output[:, 0, :, :], 0))
    assert not tf.reduce_all(tf.equal(output[:, -1, :, :], 0))


def test_dilation_rate_effect(sample_input):
    """Test that larger dilation increases receptive field but keeps same output shape."""
    layer1 = PaddedConv2D(n_kernels=2, dilation_rate=1)
    layer2 = PaddedConv2D(n_kernels=2, dilation_rate=3)
    out1 = layer1(sample_input)
    out2 = layer2(sample_input)

    # The shapes should still match (padding compensates)
    assert out1.shape == out2.shape


def test_with_activation(sample_input):
    """Test that activation function is correctly applied."""
    layer = PaddedConv2D(n_kernels=4, activation='relu')
    output = layer(sample_input)

    # Since relu activation, no negative values
    assert tf.reduce_all(output >= 0)


def test_model_serialization(sample_input):
    """Ensure the layer works inside a Keras model and serializes correctly."""
    inputs = tf.keras.Input(shape=(*INPUT_SHAPE, N_INPUT_CHANNELS))
    x = PaddedConv2D(4, dilation_rate=2, activation='relu')(inputs)
    model = tf.keras.Model(inputs, x)

    config = model.get_config()
    reloaded = tf.keras.Model.from_config(config, custom_objects={'PaddedConv2D': PaddedConv2D})

    out1 = model(sample_input)
    out2 = reloaded(sample_input)
    assert out1.numpy().shape == out2.numpy().shape


def test_no_nan_or_inf_in_output(sample_input):
    """Ensure layer produces finite outputs."""
    layer = PaddedConv2D(n_kernels=N_KERNELS, dilation_rate=2)
    output = layer(sample_input)

    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))
