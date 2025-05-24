import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported
mock_tf_convblock = MagicMock()
sys.modules['tensorflow'] = mock_tf_convblock
sys.modules['tensorflow.keras'] = mock_tf_convblock.keras
sys.modules['tensorflow.keras.layers'] = mock_tf_convblock.keras.layers

# Mock ResidualBlock (used by ConvolutionBlock)
mock_residual_block_module_convblock = MagicMock()
sys.modules['src.modules.blocks.residual_block'] = mock_residual_block_module_convblock

# Import after mocks
from modules.blocks.convolution_block import ConvolutionBlock

class TestConvolutionBlock(unittest.TestCase):

    def setUp(self):
        mock_tf_convblock.reset_mock()
        mock_residual_block_module_convblock.ResidualBlock.reset_mock() # Reset the class mock

        self.n_kernels_val = 64
        self.n_stride_val = 1 # Note: ResidualBlock in src uses stride, but ConvBlock's pooling implies stride 2 for pooling

        # Mock instances that will be created in ConvolutionBlock's __init__
        self.mock_residual_block_instance = MagicMock(name="ResidualBlockInstance")
        mock_residual_block_module_convblock.ResidualBlock.return_value = self.mock_residual_block_instance
        
        self.mock_avg_pooling_instance = MagicMock(name="AveragePoolingInstance")
        mock_tf_convblock.keras.layers.AveragePooling2D.return_value = self.mock_avg_pooling_instance

    def test_initialization(self):
        conv_block = ConvolutionBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)

        # Check parameters are stored
        self.assertEqual(conv_block.n_kernels, self.n_kernels_val)
        self.assertEqual(conv_block.n_stride, self.n_stride_val)

        # Check that ResidualBlock was instantiated correctly
        mock_residual_block_module_convblock.ResidualBlock.assert_called_once_with(
            self.n_kernels_val, self.n_stride_val
        )
        self.assertIs(conv_block.residual_block, self.mock_residual_block_instance)

        # Check that AveragePooling2D was instantiated correctly
        mock_tf_convblock.keras.layers.AveragePooling2D.assert_called_once_with(pool_size=(2, 2))
        self.assertIs(conv_block.pooling, self.mock_avg_pooling_instance)
        
        # Check that the parent Layer.__init__ was called
        mock_tf_convblock.keras.layers.Layer.assert_called_with(conv_block)


    def test_call_method(self):
        conv_block = ConvolutionBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        
        mock_input_tensor = MagicMock(name="InputTensor")
        
        # Mock the return value of the residual_block call
        mock_residual_output_tensor = MagicMock(name="ResidualOutputTensor")
        self.mock_residual_block_instance.return_value = mock_residual_output_tensor
        
        # Mock the return value of the pooling call
        mock_pooled_output_tensor = MagicMock(name="PooledOutputTensor")
        self.mock_avg_pooling_instance.return_value = mock_pooled_output_tensor

        # Call the method
        pooled_result, layer_result = conv_block.call(mock_input_tensor)

        # 1. Check ResidualBlock was called with inputs
        self.mock_residual_block_instance.assert_called_once_with(mock_input_tensor)
        
        # 2. Check AveragePooling2D was called with the output of ResidualBlock
        self.mock_avg_pooling_instance.assert_called_once_with(mock_residual_output_tensor)

        # 3. Check the returned values
        self.assertIs(pooled_result, mock_pooled_output_tensor)
        self.assertIs(layer_result, mock_residual_output_tensor)

    def test_get_config(self):
        conv_block = ConvolutionBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        
        # Mock super().get_config()
        mock_parent_config = {'name': 'convolution_block_1'}
        # To mock super().get_config(), we need to patch Layer's get_config or have Layer mocked.
        # Since Layer is already mocked (mock_tf_convblock.keras.layers.Layer),
        # we can set a return_value for its get_config method.
        mock_tf_convblock.keras.layers.Layer.return_value.get_config.return_value = mock_parent_config.copy()
        
        config = conv_block.get_config()
        
        expected_config = {
            **mock_parent_config,
            'n_kernels': self.n_kernels_val,
            'n_stride': self.n_stride_val
        }
        self.assertEqual(config, expected_config)

    @patch.object(ConvolutionBlock, '__init__', return_value=None) # Prevent __init__ during from_config direct call
    def test_from_config(self, mock_init):
        config_data = {
            'name': 'conv_block_from_config', 
            'n_kernels': 128, 
            'n_stride': 2
        }
        # We expect ConvolutionBlock to be eventually called with these config values.
        # The @classmethod means `cls` is ConvolutionBlock.
        # So, `cls(**config)` is `ConvolutionBlock(**config_data)`
        
        # To test this properly, we'd ideally want to check that an instance of 
        # ConvolutionBlock is created and that its __init__ was called with config_data.
        # However, directly calling ConvolutionBlock.from_config() will run its __init__.
        # The patch on __init__ helps isolate the from_config logic.
        # A more direct way is to call the class method and then assert __init__ was called by it.
        
        # For now, let's assume the primary goal is to see if cls(**config) is invoked.
        # We can't directly mock `cls` inside the classmethod.
        # Instead, we can call it and ensure our patched __init__ (which is now ConvolutionBlock's init)
        # was called with the right args by the from_config logic.
        
        # Reset mock_init for this specific test call via from_config
        mock_init.reset_mock() 
        
        instance = ConvolutionBlock.from_config(config_data)
        
        # Check that the __init__ of ConvolutionBlock (which is mock_init here)
        # was called with the unpacked config data.
        mock_init.assert_called_once_with(**config_data)
        # The instance returned would be None because __init__ is mocked to return None.
        # This test mainly verifies that the config is passed to the constructor.


if __name__ == '__main__':
    unittest.main()
