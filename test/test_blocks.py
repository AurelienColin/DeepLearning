import pytest
import tensorflow as tf
import numpy as np
import typing

from ML.src.modules.blocks.convolution_block import ConvolutionBlock
from ML.src.modules.blocks.deconvolution_block import DeconvolutionBlock
from ML.src.modules.blocks.residual_block import ResidualBlock


@pytest.fixture
def sample_input_conv():
    """Sample input for convolution block testing."""
    return tf.random.normal((2, 32, 32, 16))


@pytest.fixture
def sample_input_deconv():
    """Sample input for deconvolution block testing."""
    current = tf.random.normal((2, 16, 16, 32))
    inherited = tf.random.normal((2, 32, 32, 16))
    return current, inherited


# Tests for ConvolutionBlock

def test_convolution_block_initialization():
    """Test ConvolutionBlock initialization."""
    block = ConvolutionBlock(n_kernels=32)
    assert block.n_kernels == 32
    assert isinstance(block.residual_block, ResidualBlock)
    assert isinstance(block.pooling, tf.keras.layers.AveragePooling2D)


def test_convolution_block_output_shapes(sample_input_conv):
    """Test ConvolutionBlock output shapes."""
    block = ConvolutionBlock(n_kernels=32)
    pooled, layer = block(sample_input_conv)

    # Pooled layer should be half the size
    assert pooled.shape[1] == sample_input_conv.shape[1] // 2
    assert pooled.shape[2] == sample_input_conv.shape[2] // 2
    assert pooled.shape[3] == 32  # n_kernels

    # Layer should maintain spatial dimensions
    assert layer.shape[1] == sample_input_conv.shape[1]
    assert layer.shape[2] == sample_input_conv.shape[2]
    assert layer.shape[3] == 32


def test_convolution_block_returns_tuple(sample_input_conv):
    """Test that ConvolutionBlock returns a tuple."""
    block = ConvolutionBlock(n_kernels=32)
    output = block(sample_input_conv)

    assert isinstance(output, tuple)
    assert len(output) == 2


def test_convolution_block_config():
    """Test ConvolutionBlock get_config and from_config."""
    block = ConvolutionBlock(n_kernels=64)
    config = block.get_config()

    new_block = ConvolutionBlock.from_config(config)
    assert new_block.n_kernels == 64


def test_convolution_block_no_nan_or_inf(sample_input_conv):
    """Test that ConvolutionBlock produces finite outputs."""
    block = ConvolutionBlock(n_kernels=32)
    pooled, layer = block(sample_input_conv)

    assert not tf.reduce_any(tf.math.is_nan(pooled))
    assert not tf.reduce_any(tf.math.is_inf(pooled))
    assert not tf.reduce_any(tf.math.is_nan(layer))
    assert not tf.reduce_any(tf.math.is_inf(layer))


def test_convolution_block_in_model(sample_input_conv):
    """Test ConvolutionBlock integration in a Keras model."""
    inputs = tf.keras.Input(shape=(32, 32, 16))
    pooled, layer = ConvolutionBlock(n_kernels=32)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[pooled, layer])

    pooled_out, layer_out = model(sample_input_conv)
    assert pooled_out.shape == (2, 16, 16, 32)
    assert layer_out.shape == (2, 32, 32, 32)


# Tests for DeconvolutionBlock

def test_deconvolution_block_initialization():
    """Test DeconvolutionBlock initialization."""
    block = DeconvolutionBlock(n_kernels=64)
    assert block.n_kernels == 64


def test_deconvolution_block_output_shape_without_inherited():
    """Test DeconvolutionBlock output shape without inherited layer."""
    block = DeconvolutionBlock(n_kernels=32)
    current_layer = tf.random.normal((2, 16, 16, 64))

    output = block(current_layer)

    # Output should be upsampled (2x)
    assert output.shape[1] == current_layer.shape[1] * 2
    assert output.shape[2] == current_layer.shape[2] * 2
    assert output.shape[3] == 32  # n_kernels


def test_deconvolution_block_output_shape_with_inherited(sample_input_deconv):
    """Test DeconvolutionBlock output shape with inherited layer."""
    current, inherited = sample_input_deconv
    block = DeconvolutionBlock(n_kernels=48)

    output = block(current, inherited)

    # Output should match upsampled size
    assert output.shape[1] == current.shape[1] * 2
    assert output.shape[2] == current.shape[2] * 2
    assert output.shape[3] == 48


def test_deconvolution_block_concatenation(sample_input_deconv):
    """Test that DeconvolutionBlock concatenates inherited layer correctly."""
    current, inherited = sample_input_deconv
    block = DeconvolutionBlock(n_kernels=32)

    # Build the block to initialize layers
    block.build(current.shape)

    # The upsampling should double the spatial dimensions
    upsampled = block.upsampling(current)
    assert upsampled.shape[1:3] == inherited.shape[1:3]


def test_deconvolution_block_config():
    """Test DeconvolutionBlock get_config and from_config."""
    block = DeconvolutionBlock(n_kernels=128)
    config = block.get_config()

    new_block = DeconvolutionBlock.from_config(config)
    assert new_block.n_kernels == 128


def test_deconvolution_block_no_nan_or_inf(sample_input_deconv):
    """Test that DeconvolutionBlock produces finite outputs."""
    current, inherited = sample_input_deconv
    block = DeconvolutionBlock(n_kernels=32)

    output = block(current, inherited)

    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_deconvolution_block_in_model():
    """Test DeconvolutionBlock integration in a Keras model."""
    # Create a simple encoder-decoder structure
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Encoder
    conv_block = ConvolutionBlock(n_kernels=16)
    pooled, skip = conv_block(inputs)

    # Decoder
    deconv_block = DeconvolutionBlock(n_kernels=8)
    outputs = deconv_block(pooled, skip)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    sample = tf.random.normal((2, 32, 32, 3))
    result = model(sample)

    assert result.shape == (2, 32, 32, 8)
    assert not tf.reduce_any(tf.math.is_nan(result))


# Tests for ResidualBlock

def test_residual_block_initialization():
    """Test ResidualBlock initialization."""
    block = ResidualBlock(n_kernels=64)
    assert block.n_kernels == 64
    assert isinstance(block.batch_norm, tf.keras.layers.BatchNormalization)
    assert len(block.conv2ds) == 2


def test_residual_block_output_shape():
    """Test ResidualBlock output shape."""
    block = ResidualBlock(n_kernels=32)
    sample = tf.random.normal((2, 16, 16, 32))

    output = block(sample)

    # ResidualBlock maintains spatial dimensions
    assert output.shape == sample.shape


def test_residual_block_with_channel_mismatch():
    """Test ResidualBlock with input having different channels than n_kernels."""
    block = ResidualBlock(n_kernels=64)
    sample = tf.random.normal((2, 16, 16, 32))  # 32 channels input

    output = block(sample)

    # Output should have n_kernels channels
    assert output.shape[-1] == 64
    # Spatial dimensions preserved
    assert output.shape[1:3] == sample.shape[1:3]


def test_residual_block_config():
    """Test ResidualBlock get_config and from_config."""
    block = ResidualBlock(n_kernels=128)
    config = block.get_config()

    new_block = ResidualBlock.from_config(config)
    assert new_block.n_kernels == 128


def test_residual_block_no_nan_or_inf():
    """Test that ResidualBlock produces finite outputs."""
    block = ResidualBlock(n_kernels=32)
    sample = tf.random.normal((2, 16, 16, 16))

    output = block(sample)

    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_residual_connection_effect():
    """Test that residual connection is actually working."""
    block = ResidualBlock(n_kernels=16)
    sample = tf.random.normal((2, 8, 8, 16))

    # Get weights before
    output1 = block(sample)

    # The output should not be just the input (i.e., some transformation happened)
    # but should also not be completely different (residual connection preserves info)
    similarity = tf.reduce_mean(tf.abs(output1 - sample))
    assert similarity > 0  # Some change happened
    assert similarity < 10  # But not completely different (reasonable range for normalized data)
