import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported
mock_tf_resblock = MagicMock()
sys.modules['tensorflow'] = mock_tf_resblock
sys.modules['tensorflow.keras'] = mock_tf_resblock.keras
sys.modules['tensorflow.keras.layers'] = mock_tf_resblock.keras.layers

# Mock AtrousConv2D (used by ResidualBlock)
mock_atrous_conv2d_module_resblock = MagicMock()
sys.modules['src.modules.layers.atrous_conv2d'] = mock_atrous_conv2d_module_resblock

# Import after mocks
from modules.blocks.residual_block import ResidualBlock

class TestResidualBlock(unittest.TestCase):

    def setUp(self):
        mock_tf_resblock.reset_mock()
        mock_atrous_conv2d_module_resblock.AtrousConv2D.reset_mock()

        self.n_kernels_val = 64
        self.n_stride_val = 1

        # Mock instances that will be created in ResidualBlock's __init__
        self.mock_batch_norm_instance = MagicMock(name="BatchNormInstance")
        mock_tf_resblock.keras.layers.BatchNormalization.return_value = self.mock_batch_norm_instance
        
        self.mock_atrous_conv1_instance = MagicMock(name="AtrousConv1Instance")
        self.mock_atrous_conv2_instance = MagicMock(name="AtrousConv2Instance")
        # AtrousConv2D is called twice in __init__
        mock_atrous_conv2d_module_resblock.AtrousConv2D.side_effect = [
            self.mock_atrous_conv1_instance, self.mock_atrous_conv2_instance
        ]
        
        self.mock_add_instance = MagicMock(name="AddInstance")
        mock_tf_resblock.keras.layers.Add.return_value = self.mock_add_instance

        # Mock for Conv2D created in build() if needed
        self.mock_residual_conv_instance = MagicMock(name="ResidualConvInstance")
        # We'll make this the return value when tf.keras.layers.Conv2D is called
        mock_tf_resblock.keras.layers.Conv2D.return_value = self.mock_residual_conv_instance


    def test_initialization(self):
        res_block = ResidualBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)

        self.assertEqual(res_block.n_kernels, self.n_kernels_val)
        self.assertEqual(res_block.n_stride, self.n_stride_val)
        self.assertIsNone(res_block.residual_conv) # Not built yet

        # Check BatchNormalization instantiation
        mock_tf_resblock.keras.layers.BatchNormalization.assert_called_once_with()
        self.assertIs(res_block.batch_norm, self.mock_batch_norm_instance)

        # Check AtrousConv2D instantiations
        expected_atrous_calls = [
            call(self.n_kernels_val, activation='swish', n_stride=self.n_stride_val),
            call(self.n_kernels_val, activation=None, n_stride=self.n_stride_val)
        ]
        mock_atrous_conv2d_module_resblock.AtrousConv2D.assert_has_calls(expected_atrous_calls)
        self.assertEqual(mock_atrous_conv2d_module_resblock.AtrousConv2D.call_count, 2)
        self.assertEqual(len(res_block.atrous_conv2ds), 2)
        self.assertIs(res_block.atrous_conv2ds[0], self.mock_atrous_conv1_instance)
        self.assertIs(res_block.atrous_conv2ds[1], self.mock_atrous_conv2_instance)

        # Check Add instantiation
        mock_tf_resblock.keras.layers.Add.assert_called_once_with()
        self.assertIs(res_block.add, self.mock_add_instance)

        # Check parent Layer.__init__ call
        mock_tf_resblock.keras.layers.Layer.assert_called_with(res_block)

    def test_build_with_matching_channels(self):
        res_block = ResidualBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        # Input shape: (batch, height, width, channels)
        # Channels match n_kernels, so residual_conv should NOT be created.
        input_shape_matching = (None, 32, 32, self.n_kernels_val) 
        
        res_block.build(input_shape_matching)
        
        self.assertIsNone(res_block.residual_conv)
        mock_tf_resblock.keras.layers.Conv2D.assert_not_called()

    def test_build_with_different_channels(self):
        res_block = ResidualBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        # Channels (self.n_kernels_val - 1) do NOT match n_kernels.
        input_shape_different = (None, 32, 32, self.n_kernels_val - 1)
        
        res_block.build(input_shape_different)
        
        self.assertIsNotNone(res_block.residual_conv)
        mock_tf_resblock.keras.layers.Conv2D.assert_called_once_with(
            self.n_kernels_val, kernel_size=(1,1)
        )
        self.assertIs(res_block.residual_conv, self.mock_residual_conv_instance)


    def test_call_method_no_residual_conv(self):
        # Case where input channels == n_kernels, so residual_conv is None
        res_block = ResidualBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        res_block.residual_conv = None # Explicitly ensure it's None for this test path

        mock_input_tensor = MagicMock(name="InputTensorNoResConv")
        
        # Mock outputs of layers in the main path
        mock_bn_output = MagicMock(name="BNOutput")
        self.mock_batch_norm_instance.return_value = mock_bn_output
        
        mock_ac1_output = MagicMock(name="AtrousConv1Output")
        self.mock_atrous_conv1_instance.return_value = mock_ac1_output
        
        mock_ac2_output = MagicMock(name="AtrousConv2Output") # This is 'x' before Add
        self.mock_atrous_conv2_instance.return_value = mock_ac2_output
        
        mock_add_final_output = MagicMock(name="AddFinalOutput")
        self.mock_add_instance.return_value = mock_add_final_output

        # Call the method
        result = res_block.call(mock_input_tensor)

        # 1. residual_conv path: not called, residual is inputs
        # (self.mock_residual_conv_instance is not involved)

        # 2. Main path calls
        self.mock_batch_norm_instance.assert_called_once_with(mock_input_tensor)
        self.mock_atrous_conv1_instance.assert_called_once_with(mock_bn_output)
        self.mock_atrous_conv2_instance.assert_called_once_with(mock_ac1_output)
        
        # 3. Add layer called with [x, residual] = [mock_ac2_output, mock_input_tensor]
        self.mock_add_instance.assert_called_once_with([mock_ac2_output, mock_input_tensor])
        
        # 4. Check result
        self.assertIs(result, mock_add_final_output)


    def test_call_method_with_residual_conv(self):
        # Case where input channels != n_kernels, so residual_conv is created and used
        res_block = ResidualBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        # Simulate that build() created residual_conv
        res_block.residual_conv = self.mock_residual_conv_instance 
        
        mock_input_tensor = MagicMock(name="InputTensorWithResConv")
        mock_projected_residual = MagicMock(name="ProjectedResidual")
        self.mock_residual_conv_instance.return_value = mock_projected_residual
        
        mock_bn_output = MagicMock(name="BNOutputWRC")
        self.mock_batch_norm_instance.return_value = mock_bn_output
        mock_ac1_output = MagicMock(name="AtrousConv1OutputWRC")
        self.mock_atrous_conv1_instance.return_value = mock_ac1_output
        mock_ac2_output = MagicMock(name="AtrousConv2OutputWRC")
        self.mock_atrous_conv2_instance.return_value = mock_ac2_output
        mock_add_final_output = MagicMock(name="AddFinalOutputWRC")
        self.mock_add_instance.return_value = mock_add_final_output

        result = res_block.call(mock_input_tensor)

        # 1. residual_conv path: called with inputs
        self.mock_residual_conv_instance.assert_called_once_with(mock_input_tensor)
        # (residual is mock_projected_residual)

        # 2. Main path calls (same as before)
        self.mock_batch_norm_instance.assert_called_once_with(mock_input_tensor)
        self.mock_atrous_conv1_instance.assert_called_once_with(mock_bn_output)
        self.mock_atrous_conv2_instance.assert_called_once_with(mock_ac1_output)
        
        # 3. Add layer called with [x, residual] = [mock_ac2_output, mock_projected_residual]
        self.mock_add_instance.assert_called_once_with([mock_ac2_output, mock_projected_residual])
        
        self.assertIs(result, mock_add_final_output)


    def test_get_config(self):
        res_block = ResidualBlock(n_kernels=self.n_kernels_val, n_stride=self.n_stride_val)
        mock_parent_config = {'name': 'residual_block_1'}
        mock_tf_resblock.keras.layers.Layer.return_value.get_config.return_value = mock_parent_config.copy()
        
        config = res_block.get_config()
        
        expected_config = {
            **mock_parent_config,
            'n_kernels': self.n_kernels_val,
            'n_stride': self.n_stride_val
        }
        self.assertEqual(config, expected_config)

    @patch.object(ResidualBlock, '__init__', return_value=None)
    def test_from_config(self, mock_init):
        config_data = {
            'name': 'res_block_from_config', 
            'n_kernels': 48, 
            'n_stride': 2
        }
        mock_init.reset_mock()
        
        instance = ResidualBlock.from_config(config_data)
        
        mock_init.assert_called_once_with(**config_data)


if __name__ == '__main__':
    unittest.main()
