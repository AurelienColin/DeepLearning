import unittest
import tensorflow as tf
import typing

from src.modules.blocks.deconvolution_block import DeconvolutionBlock
# from src.modules.blocks.residual_block import ResidualBlock # Implicitly used

class TestDeconvolutionBlock(unittest.TestCase):
    def test_instantiation(self) -> None:
        deconv_block = DeconvolutionBlock(n_kernels=32, n_stride=1)
        self.assertIsInstance(deconv_block, DeconvolutionBlock)

    def test_call_output_shape_with_inherited_layer(self) -> None:
        batch_size: int = 2
        height: int = 16 # Height after downsampling
        width: int = 16  # Width after downsampling
        channels_current: int = 32 # Kernels from previous layer
        channels_inherited: int = 16 # Kernels from corresponding encoder layer
        n_kernels: int = 16 # Target kernels for this deconv block
        internal_stride: int = 1
        current_layer_tensor: tf.Tensor = tf.random.normal([batch_size, height, width, channels_current])
        
        deconv_block_with_skip = DeconvolutionBlock(n_kernels=n_kernels, n_stride=internal_stride)
        inherited_layer_tensor: tf.Tensor = tf.random.normal([batch_size, height * 2, width * 2, channels_inherited])
        output_tensor_with_skip = deconv_block_with_skip(current_layer_tensor, inherited_layer_tensor)
        
        self.assertIsInstance(output_tensor_with_skip, tf.Tensor)
        expected_height_width = height * 2
        self.assertEqual(output_tensor_with_skip.shape, (batch_size, expected_height_width, expected_height_width, n_kernels))

    def test_call_output_shape_without_inherited_layer(self) -> None:
        batch_size: int = 2
        height: int = 16 # Height after downsampling
        width: int = 16  # Width after downsampling
        channels_current: int = 32 # Kernels from previous layer
        n_kernels: int = 16 # Target kernels for this deconv block
        internal_stride: int = 1
        current_layer_tensor: tf.Tensor = tf.random.normal([batch_size, height, width, channels_current])

        deconv_block_no_skip = DeconvolutionBlock(n_kernels=n_kernels, n_stride=internal_stride)
        output_tensor_no_skip = deconv_block_no_skip(current_layer_tensor, None)
        self.assertIsInstance(output_tensor_no_skip, tf.Tensor)
        expected_height_width = height * 2
        self.assertEqual(output_tensor_no_skip.shape, (batch_size, expected_height_width, expected_height_width, n_kernels))

if __name__ == '__main__':
    unittest.main()
