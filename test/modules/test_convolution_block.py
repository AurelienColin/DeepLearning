import pytest # Added for consistency
import tensorflow as tf
import typing

from src.modules.blocks.convolution_block import ConvolutionBlock
# TestConvolutionBlock also uses ResidualBlock implicitly, so it might be good to have it,
# but the test only asserts properties of ConvolutionBlock's direct output.
# from src.modules.blocks.residual_block import ResidualBlock

# Removed TestConvolutionBlock class wrapper, tests are now functions

def test_instantiation() -> None:
    conv_block = ConvolutionBlock(n_kernels=32, n_stride=1)
    assert isinstance(conv_block, ConvolutionBlock)

def test_call_and_output_shape() -> None:
    batch_size: int = 2
    height: int = 32
    width: int = 32
    channels: int = 3
    input_tensor: tf.Tensor = tf.random.normal([batch_size, height, width, channels])

    n_kernels: int = 16
    # n_stride for AtrousConv2D within ResidualBlock, not direct stride of ConvBlock's pooling
    internal_stride: int = 1

    conv_block = ConvolutionBlock(n_kernels=n_kernels, n_stride=internal_stride)
    pooled_output, direct_output = conv_block(input_tensor)

    assert isinstance(pooled_output, tf.Tensor)
    assert isinstance(direct_output, tf.Tensor)

    # direct_output shape is after ResidualBlock, before pooling
    assert direct_output.shape == (batch_size, height, width, n_kernels)
    # pooled_output shape is after AveragePooling2D
    assert pooled_output.shape == (batch_size, height // 2, width // 2, n_kernels)
