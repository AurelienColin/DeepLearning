import unittest
import tensorflow as tf
import typing

from src.modules.blocks.convolution_block import ConvolutionBlock
# TestConvolutionBlock also uses ResidualBlock implicitly, so it might be good to have it,
# but the test only asserts properties of ConvolutionBlock's direct output.
# from src.modules.blocks.residual_block import ResidualBlock

class TestConvolutionBlock(unittest.TestCase):
    def test_instantiation(self) -> None:
        conv_block = ConvolutionBlock(n_kernels=32, n_stride=1)
        self.assertIsInstance(conv_block, ConvolutionBlock)

    def test_call_and_output_shape(self) -> None:
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
        
        self.assertIsInstance(pooled_output, tf.Tensor)
        self.assertIsInstance(direct_output, tf.Tensor)
        
        # direct_output shape is after ResidualBlock, before pooling
        self.assertEqual(direct_output.shape, (batch_size, height, width, n_kernels))
        # pooled_output shape is after AveragePooling2D
        self.assertEqual(pooled_output.shape, (batch_size, height // 2, width // 2, n_kernels))

if __name__ == '__main__':
    unittest.main()
