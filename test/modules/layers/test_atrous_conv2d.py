import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported
mock_tf_atrous = MagicMock()
sys.modules['tensorflow'] = mock_tf_atrous
sys.modules['tensorflow.keras'] = mock_tf_atrous.keras
sys.modules['tensorflow.keras.layers'] = mock_tf_atrous.keras.layers

# Mock PaddedConv2D (used by AtrousConv2D)
mock_padded_conv2d_module_atrous = MagicMock()
sys.modules['src.modules.layers.padded_conv2d'] = mock_padded_conv2d_module_atrous

# Mock numpy (used in build method)
mock_numpy_atrous = MagicMock()
sys.modules['numpy'] = mock_numpy_atrous

# Import after mocks
from modules.layers.atrous_conv2d import AtrousConv2D

class TestAtrousConv2D(unittest.TestCase):

    def setUp(self):
        mock_tf_atrous.reset_mock()
        mock_padded_conv2d_module_atrous.PaddedConv2D.reset_mock()
        mock_numpy_atrous.reset_mock()

        self.n_kernels_val = 64
        self.n_stride_val = 2 # Results in 2 PaddedConv2D layers
        self.activation_val = 'relu'
        self.input_shape_val = (None, 32, 32, 3) # Batch, H, W, C

        # Mock for Concatenate layer created in call()
        self.mock_concatenate_instance = MagicMock(name="ConcatenateInstance")
        mock_tf_atrous.keras.layers.Concatenate.return_value = self.mock_concatenate_instance


    def test_initialization(self):
        atrous_layer = AtrousConv2D(
            n_kernels=self.n_kernels_val,
            n_stride=self.n_stride_val,
            activation=self.activation_val,
            name="test_atrous"
        )

        self.assertEqual(atrous_layer.n_kernels, self.n_kernels_val)
        self.assertEqual(atrous_layer.n_stride, self.n_stride_val)
        self.assertEqual(atrous_layer.activation, self.activation_val)
        self.assertIsNone(atrous_layer.conv_layers) # Not built yet
        
        # Check parent Layer.__init__ was called (name is a good proxy)
        mock_tf_atrous.keras.layers.Layer.assert_called_with(atrous_layer, name="test_atrous")

    def test_build_method_stride_gt_1(self):
        atrous_layer = AtrousConv2D(
            n_kernels=self.n_kernels_val, # 64
            n_stride=self.n_stride_val,   # 2
            activation=self.activation_val
        )

        # Mock numpy.linspace and np.ediff1d
        # linspace(0, 64, 2+1) = linspace(0, 64, 3) -> e.g., [0, 32, 64]
        # ediff1d([0, 32, 64]) -> [32, 32] (these are kernel_counts for each PaddedConv2D)
        mock_numpy_atrous.linspace.return_value = MagicMock(name="linspace_ret")
        mock_numpy_atrous.linspace.return_value.astype.return_value = [0, 32, 64] # Example
        mock_numpy_atrous.ediff1d.return_value = [32, 32] # kernel_counts

        # Mock PaddedConv2D instances that will be created
        mock_padded_conv1 = MagicMock(name="PaddedConv1")
        mock_padded_conv2 = MagicMock(name="PaddedConv2")
        mock_padded_conv2d_module_atrous.PaddedConv2D.side_effect = [mock_padded_conv1, mock_padded_conv2]

        atrous_layer.build(input_shape=self.input_shape_val)

        mock_numpy_atrous.linspace.assert_called_once_with(0, self.n_kernels_val, self.n_stride_val + 1)
        mock_numpy_atrous.linspace.return_value.astype.assert_called_once_with(int)
        mock_numpy_atrous.ediff1d.assert_called_once_with([0, 32, 64])

        expected_padded_conv_calls = [
            call(activation=self.activation_val, n_kernels=32, dilation_rate=1, name="padded_conv_dilation_1"),
            call(activation=self.activation_val, n_kernels=32, dilation_rate=2, name="padded_conv_dilation_2")
        ]
        mock_padded_conv2d_module_atrous.PaddedConv2D.assert_has_calls(expected_padded_conv_calls)
        self.assertEqual(mock_padded_conv2d_module_atrous.PaddedConv2D.call_count, 2)
        
        self.assertEqual(len(atrous_layer.conv_layers), 2)
        self.assertIs(atrous_layer.conv_layers[0], mock_padded_conv1)
        self.assertIs(atrous_layer.conv_layers[1], mock_padded_conv2)
        
        # Check super().build() was called
        mock_tf_atrous.keras.layers.Layer.return_value.build.assert_called_with(atrous_layer, self.input_shape_val)


    def test_build_method_stride_1(self):
        n_stride_one = 1
        atrous_layer = AtrousConv2D(
            n_kernels=self.n_kernels_val, # 64
            n_stride=n_stride_one,      # 1
            activation=self.activation_val
        )
        # kernel_counts should be [self.n_kernels] = [64]
        mock_padded_conv_s1 = MagicMock(name="PaddedConv_s1")
        mock_padded_conv2d_module_atrous.PaddedConv2D.return_value = mock_padded_conv_s1

        atrous_layer.build(input_shape=self.input_shape_val)

        mock_numpy_atrous.linspace.assert_not_called() # Not called if n_stride == 1
        mock_numpy_atrous.ediff1d.assert_not_called()

        mock_padded_conv2d_module_atrous.PaddedConv2D.assert_called_once_with(
            activation=self.activation_val, 
            n_kernels=self.n_kernels_val, # All kernels to one layer
            dilation_rate=1,             # Dilation rate starts at 1
            name="padded_conv_dilation_1"
        )
        self.assertEqual(len(atrous_layer.conv_layers), 1)
        self.assertIs(atrous_layer.conv_layers[0], mock_padded_conv_s1)


    def test_call_method(self):
        atrous_layer = AtrousConv2D(
            n_kernels=self.n_kernels_val, n_stride=self.n_stride_val, activation=self.activation_val
        )
        # Manually build the layer to populate self.conv_layers
        mock_padded_conv1 = MagicMock(name="PaddedConv1_call")
        mock_padded_conv2 = MagicMock(name="PaddedConv2_call")
        atrous_layer.conv_layers = [mock_padded_conv1, mock_padded_conv2] # Simulate build()
        
        mock_input_tensor = MagicMock(name="InputTensorForAtrousCall")
        
        # Mock outputs from each PaddedConv2D layer
        mock_output_conv1 = MagicMock(name="OutputConv1")
        mock_output_conv2 = MagicMock(name="OutputConv2")
        mock_padded_conv1.return_value = mock_output_conv1
        mock_padded_conv2.return_value = mock_output_conv2
        
        # Mock output from Concatenate layer
        mock_final_concatenated_output = MagicMock(name="FinalConcatenatedOutput")
        self.mock_concatenate_instance.return_value = mock_final_concatenated_output

        result = atrous_layer.call(mock_input_tensor)

        # Check that each PaddedConv2D layer was called with the input tensor
        mock_padded_conv1.assert_called_once_with(mock_input_tensor)
        mock_padded_conv2.assert_called_once_with(mock_input_tensor)
        
        # Check that Concatenate was called with the list of outputs
        mock_tf_atrous.keras.layers.Concatenate.assert_called_once_with() # Instantiation
        self.mock_concatenate_instance.assert_called_once_with([mock_output_conv1, mock_output_conv2]) # Call
        
        self.assertIs(result, mock_final_concatenated_output)


    def test_get_config(self):
        atrous_layer = AtrousConv2D(
            n_kernels=self.n_kernels_val, 
            n_stride=self.n_stride_val, 
            activation=self.activation_val,
            name="get_config_atrous"
        )
        mock_parent_config = {'name': 'get_config_atrous', 'trainable': True} # Example
        mock_tf_atrous.keras.layers.Layer.return_value.get_config.return_value = mock_parent_config.copy()

        config = atrous_layer.get_config()

        expected_config = {
            **mock_parent_config,
            'n_kernels': self.n_kernels_val,
            'activation': self.activation_val,
            'n_stride': self.n_stride_val
        }
        self.assertEqual(config, expected_config)

    @patch.object(AtrousConv2D, '__init__', return_value=None)
    def test_from_config(self, mock_init):
        config_data = {
            'name': 'from_config_atrous', 
            'n_kernels': 32, 
            'n_stride': 1,
            'activation': 'sigmoid'
        }
        mock_init.reset_mock()
        
        instance = AtrousConv2D.from_config(config_data)
        
        mock_init.assert_called_once_with(**config_data)


if __name__ == '__main__':
    unittest.main()
