import typing
import pytest
from ML.src.modules.layers.atrous_conv2d import AtrousConv2D
from ML.src.modules.layers.padded_conv2d import PaddedConv2D
import tensorflow as tf
import numpy as np

BATCH_SIZE: int = 4
INPUT_SHAPE: typing.Tuple[int, int] = (16, 24)
N_KERNELS: int = 15
N_INPUT_CHANNELS: int = 2
N_STRIDE: int = 3


@pytest.fixture
def sample_input() -> tf.Tensor:
    """Shape: (batch, height, width, channels)"""
    return tf.random.normal((BATCH_SIZE, *INPUT_SHAPE, N_INPUT_CHANNELS))


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


# Enhanced tests inspired by test_from_another_repo

def test_initialization() -> None:
    """Test that layer properties are correctly initialized."""
    layer = AtrousConv2D(n_kernels=8, n_stride=N_STRIDE, activation='relu')
    assert layer.n_kernels == 8
    assert layer.n_stride == N_STRIDE
    assert layer.activation == 'relu'
    assert layer.conv_layers is None  # Not built yet


def test_build_creates_correct_layers(sample_input: tf.Tensor) -> None:
    """Test that build creates the correct number of PaddedConv2D layers."""
    layer = AtrousConv2D(n_kernels=N_KERNELS, n_stride=N_STRIDE)
    layer.build(sample_input.shape)

    assert isinstance(layer.conv_layers, list)
    assert len(layer.conv_layers) == N_STRIDE
    for conv in layer.conv_layers:
        assert isinstance(conv, PaddedConv2D)


def test_call_outputs_concatenated(sample_input: tf.Tensor) -> None:
    """Test that the layer concatenates outputs from all dilation rates."""
    layer = AtrousConv2D(n_kernels=N_KERNELS, n_stride=N_STRIDE)
    output = layer(sample_input)

    assert isinstance(output, tf.Tensor)
    assert output.shape[-1] == N_KERNELS
    assert output.shape[:-1] == sample_input.shape[:-1]


def test_get_and_from_config() -> None:
    """Test serialization and deserialization."""
    layer = AtrousConv2D(n_kernels=12, n_stride=4, activation='sigmoid')
    sample = tf.random.normal((2, 16, 16, 3))
    layer(sample)  # Build the layer

    config = layer.get_config()
    new_layer = AtrousConv2D.from_config(config)

    assert new_layer.n_kernels == 12
    assert new_layer.n_stride == 4
    assert new_layer.activation == 'sigmoid'


def test_model_integration(sample_input: tf.Tensor) -> None:
    """Ensure it works within a compiled Keras model."""
    inputs = tf.keras.Input(shape=(*INPUT_SHAPE, N_INPUT_CHANNELS))
    outputs = AtrousConv2D(N_KERNELS, n_stride=N_STRIDE, activation='relu')(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    result = model(sample_input)
    assert result.shape[-1] == N_KERNELS
    assert not tf.math.reduce_any(tf.math.is_nan(result))
    assert not tf.math.reduce_any(tf.math.is_inf(result))


def test_serialization(sample_input: tf.Tensor) -> None:
    """Test model serialization with AtrousConv2D layer."""
    inputs = tf.keras.Input(shape=(*INPUT_SHAPE, N_INPUT_CHANNELS))
    outputs = AtrousConv2D(N_KERNELS, n_stride=N_STRIDE, activation='relu')(inputs)
    model = tf.keras.Model(inputs, outputs)

    config = model.get_config()
    reloaded = tf.keras.Model.from_config(config, custom_objects={
        'AtrousConv2D': AtrousConv2D,
        'PaddedConv2D': PaddedConv2D,
    })

    out1 = model(sample_input)
    out2 = reloaded(sample_input)
    assert out1.numpy().shape == out2.numpy().shape


def test_with_activation(sample_input: tf.Tensor) -> None:
    """Test that activation function is correctly applied."""
    layer = AtrousConv2D(n_kernels=N_KERNELS, n_stride=N_STRIDE, activation='relu')
    output = layer(sample_input)

    # Since relu activation, no negative values
    assert tf.reduce_all(output >= 0)


def test_varying_dilation_rates(sample_input: tf.Tensor) -> None:
    """Test that different n_stride values produce expected output channels."""
    for n_stride in [1, 2, 4]:
        layer = AtrousConv2D(n_kernels=12, n_stride=n_stride)
        output = layer(sample_input)
        assert output.shape[-1] == 12
        assert len(layer.conv_layers) == n_stride
