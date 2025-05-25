import unittest
import tensorflow as tf
import typing

from src.modules.blocks.residual_block import ResidualBlock

class TestResidualBlock(unittest.TestCase):
    def test_instantiation(self) -> None:
        residual_block_1 = ResidualBlock(n_kernels=32, n_stride=1)
        self.assertIsInstance(residual_block_1, ResidualBlock)
        residual_block_2 = ResidualBlock(n_kernels=64, n_stride=2)
        self.assertIsInstance(residual_block_2, ResidualBlock)

    def test_call_output_shape_same_channels(self) -> None:
        batch_size: int = 2
        height: int = 32
        width: int = 32
        input_channels: int = 3
        # n_kernels will be same as input_channels for this test
        n_stride: int = 1
        input_tensor: tf.Tensor = tf.random.normal([batch_size, height, width, input_channels])
        
        residual_block_same_channels = ResidualBlock(n_kernels=input_channels, n_stride=n_stride)
        output_tensor_same = residual_block_same_channels(input_tensor)
        self.assertEqual(output_tensor_same.shape, (batch_size, height, width, input_channels))

    def test_call_output_shape_diff_channels(self) -> None:
        batch_size: int = 2
        height: int = 32
        width: int = 32
        input_channels: int = 3
        n_kernels: int = 16 # Different from input_channels
        n_stride: int = 1
        input_tensor: tf.Tensor = tf.random.normal([batch_size, height, width, input_channels])

        residual_block_diff_channels = ResidualBlock(n_kernels=n_kernels, n_stride=n_stride)
        # Build the layer to initialize residual_conv if necessary
        residual_block_diff_channels.build(input_tensor.shape)
        self.assertIsNotNone(residual_block_diff_channels.residual_conv)
        output_tensor_diff = residual_block_diff_channels(input_tensor)
        self.assertEqual(output_tensor_diff.shape, (batch_size, height, width, n_kernels))

    def test_build_residual_conv_not_created_matching_channels(self) -> None:
        batch_size: int = 2
        height: int = 32
        width: int = 32
        input_channels_matching: int = 16
        n_kernels: int = 16 # Same as input_channels_matching
        n_stride: int = 1

        input_tensor_matching: tf.Tensor = tf.random.normal([batch_size, height, width, input_channels_matching])
        residual_block_matching = ResidualBlock(n_kernels=n_kernels, n_stride=n_stride)
        residual_block_matching.build(input_tensor_matching.shape)
        self.assertIsNone(residual_block_matching.residual_conv)
        _ = residual_block_matching(input_tensor_matching) # Call to ensure full build completion

    def test_build_residual_conv_created_different_channels(self) -> None:
        batch_size: int = 2
        height: int = 32
        width: int = 32
        input_channels_different: int = 3
        n_kernels: int = 16 # Different from input_channels_different
        n_stride: int = 1
        
        input_tensor_different: tf.Tensor = tf.random.normal([batch_size, height, width, input_channels_different])
        residual_block_different = ResidualBlock(n_kernels=n_kernels, n_stride=n_stride)
        residual_block_different.build(input_tensor_different.shape)
        self.assertIsNotNone(residual_block_different.residual_conv)
        self.assertIsInstance(residual_block_different.residual_conv, tf.keras.layers.Conv2D)
        _ = residual_block_different(input_tensor_different) # Call to ensure full build completion

if __name__ == '__main__':
    unittest.main()
