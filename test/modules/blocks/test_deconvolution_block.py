import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported
mock_tf_deconvblock = MagicMock()
sys.modules['tensorflow'] = mock_tf_deconvblock
sys.modules['tensorflow.keras'] = mock_tf_deconvblock.keras
sys.modules['tensorflow.keras.layers'] = mock_tf_deconvblock.keras.layers

# Mock ResidualBlock (used by DeconvolutionBlock)
mock_residual_block_module_deconvblock = MagicMock()
sys.modules['src.modules.blocks.residual_block'] = mock_residual_block_module_deconvblock

# Import after mocks
from modules.blocks.deconvolution_block import DeconvolutionBlock

class TestDeconvolutionBlock(unittest.TestCase):

    def setUp(self):
        mock_tf_deconvblock.reset_mock()
        mock_residual_block_module_deconvblock.ResidualBlock.reset_mock()

        self.n_kernels_val = 32
        self.n_stride_val = 1 # For ResidualBlock, actual upsampling is by UpSampling2D

        # Mock instances that will be created in DeconvolutionBlock's __init__
        self.mock_upsampling_instance = MagicMock(name="UpSamplingInstance")
        mock_tf_deconvblock.keras.layers.UpSampling2D.return_value = self.mock_upsampling_instance
        
        self.mock_concatenate_instance = MagicMock(name="ConcatenateInstance")
        mock_tf_deconvblock.keras.layers.Concatenate.return_value = self.mock_concatenate_instance
        
        self.mock_residual_block_instance = MagicMock(name="ResidualBlockInstanceDeconv")
        mock_residual_block_module_deconvblock.ResidualBlock.return_value = self.mock_residual_block_instance


    def test_initialization(self):
        deconv_block = DeconvolutionBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)

        self.assertEqual(deconv_block.n_kernels, self.n_kernels_val)
        self.assertEqual(deconv_block.n_stride, self.n_stride_val)

        # Check UpSampling2D instantiation
        mock_tf_deconvblock.keras.layers.UpSampling2D.assert_called_once_with(size=(2,2), interpolation="bilinear")
        self.assertIs(deconv_block.upsampling, self.mock_upsampling_instance)

        # Check Concatenate instantiation
        mock_tf_deconvblock.keras.layers.Concatenate.assert_called_once_with()
        self.assertIs(deconv_block.concat, self.mock_concatenate_instance)

        # Check ResidualBlock instantiation
        mock_residual_block_module_deconvblock.ResidualBlock.assert_called_once_with(
            self.n_kernels_val, self.n_stride_val
        )
        self.assertIs(deconv_block.residual_block, self.mock_residual_block_instance)
        
        # Check parent Layer.__init__ call
        mock_tf_deconvblock.keras.layers.Layer.assert_called_with(deconv_block)


    def test_call_method_with_inherited_layer(self):
        deconv_block = DeconvolutionBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        
        mock_current_layer_input = MagicMock(name="CurrentLayerInput")
        mock_inherited_layer_input = MagicMock(name="InheritedLayerInput")
        
        # Mock return values from layers
        mock_upsampled_output = MagicMock(name="UpsampledOutput")
        self.mock_upsampling_instance.return_value = mock_upsampled_output
        
        mock_concatenated_output = MagicMock(name="ConcatenatedOutput")
        self.mock_concatenate_instance.return_value = mock_concatenated_output
        
        mock_residual_final_output = MagicMock(name="ResidualFinalOutput")
        self.mock_residual_block_instance.return_value = mock_residual_final_output

        # Call the method
        result = deconv_block.call(mock_current_layer_input, mock_inherited_layer_input)

        # 1. UpSampling called with current_layer_input
        self.mock_upsampling_instance.assert_called_once_with(mock_current_layer_input)
        
        # 2. Concatenate called with [upsampled_output, inherited_layer_input]
        self.mock_concatenate_instance.assert_called_once_with([mock_upsampled_output, mock_inherited_layer_input])
        
        # 3. ResidualBlock called with concatenated_output
        self.mock_residual_block_instance.assert_called_once_with(mock_concatenated_output)
        
        # 4. Check result
        self.assertIs(result, mock_residual_final_output)


    def test_call_method_without_inherited_layer(self):
        deconv_block = DeconvolutionBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        
        mock_current_layer_input = MagicMock(name="CurrentLayerInputNoInherit")
        
        mock_upsampled_output = MagicMock(name="UpsampledOutputNoInherit")
        self.mock_upsampling_instance.return_value = mock_upsampled_output
        
        mock_residual_final_output = MagicMock(name="ResidualFinalOutputNoInherit")
        self.mock_residual_block_instance.return_value = mock_residual_final_output

        # Call the method with inherited_layer=None (default)
        result = deconv_block.call(mock_current_layer_input) # Or inherited_layer=None

        # 1. UpSampling called
        self.mock_upsampling_instance.assert_called_once_with(mock_current_layer_input)
        
        # 2. Concatenate should NOT be called
        self.mock_concatenate_instance.assert_not_called()
        
        # 3. ResidualBlock called with upsampled_output (since concat didn't happen)
        self.mock_residual_block_instance.assert_called_once_with(mock_upsampled_output)
        
        # 4. Check result
        self.assertIs(result, mock_residual_final_output)


    def test_get_config(self):
        deconv_block = DeconvolutionBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        mock_parent_config = {'name': 'deconvolution_block_1'}
        mock_tf_deconvblock.keras.layers.Layer.return_value.get_config.return_value = mock_parent_config.copy()
        
        config = deconv_block.get_config()
        
        expected_config = {
            **mock_parent_config,
            'n_kernels': self.n_kernels_val,
            'n_stride': self.n_stride_val
        }
        self.assertEqual(config, expected_config)

    @patch.object(DeconvolutionBlock, '__init__', return_value=None)
    def test_from_config(self, mock_init):
        config_data = {
            'name': 'deconv_block_from_config', 
            'n_kernels': 96, 
            'n_stride': 1
        }
        mock_init.reset_mock()
        
        instance = DeconvolutionBlock.from_config(config_data)
        
        mock_init.assert_called_once_with(**config_data)


if __name__ == '__main__':
    unittest.main()
