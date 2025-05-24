import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported
mock_tf_padded = MagicMock()
sys.modules['tensorflow'] = mock_tf_padded
sys.modules['tensorflow.keras'] = mock_tf_padded.keras
sys.modules['tensorflow.keras.layers'] = mock_tf_padded.keras.layers
sys.modules['tensorflow.keras.constraints'] = mock_tf_padded.keras.constraints

# Import after mocks
from modules.layers.padded_conv2d import PaddedConv2D

class TestPaddedConv2D(unittest.TestCase):

    def setUp(self):
        mock_tf_padded.reset_mock()
        
        self.n_kernels_val = 32
        self.dilation_rate_val = 2
        self.activation_val = 'relu'

        # Mock instance created in PaddedConv2D's __init__ (the Conv2D layer)
        self.mock_conv2d_internal_instance = MagicMock(name="InternalConv2DInstance")
        mock_tf_padded.keras.layers.Conv2D.return_value = self.mock_conv2d_internal_instance
        
        # Mock constraint function
        self.mock_max_norm_instance = MagicMock(name="MaxNormInstance")
        mock_tf_padded.keras.constraints.max_norm.return_value = self.mock_max_norm_instance


    def test_initialization(self):
        padded_layer = PaddedConv2D(
            n_kernels=self.n_kernels_val,
            dilation_rate=self.dilation_rate_val,
            activation=self.activation_val,
            name="test_padded_conv"
        )

        self.assertEqual(padded_layer.n_kernels, self.n_kernels_val)
        self.assertEqual(padded_layer.dilation_rate, self.dilation_rate_val)
        self.assertEqual(padded_layer.activation, self.activation_val)
        self.assertEqual(padded_layer.pad, (self.dilation_rate_val, self.dilation_rate_val))

        # Check Conv2D instantiation
        mock_tf_padded.keras.constraints.max_norm.assert_called_once_with(2.0)
        mock_tf_padded.keras.layers.Conv2D.assert_called_once_with(
            self.n_kernels_val,
            (3,3), # kernel_size
            dilation_rate=self.dilation_rate_val,
            kernel_constraint=self.mock_max_norm_instance,
            padding='valid', # Crucial for this layer's logic
            activation=self.activation_val
        )
        self.assertIs(padded_layer.conv_layer, self.mock_conv2d_internal_instance)
        
        # Check parent Layer.__init__ call
        mock_tf_padded.keras.layers.Layer.assert_called_with(padded_layer, name="test_padded_conv")


    def test_call_method(self):
        padded_layer = PaddedConv2D(
            n_kernels=self.n_kernels_val,
            dilation_rate=self.dilation_rate_val,
            activation=self.activation_val
        )
        
        mock_input_tensor = MagicMock(name="InputTensorForPaddedConv")
        
        # Mock output from the internal Conv2D layer
        mock_conv_output_tensor = MagicMock(name="ConvOutputTensor")
        self.mock_conv2d_internal_instance.return_value = mock_conv_output_tensor
        
        # Mock output from tf.pad
        mock_padded_final_output = MagicMock(name="PaddedFinalOutput")
        mock_tf_padded.pad.return_value = mock_padded_final_output

        result = padded_layer.call(mock_input_tensor)

        # 1. Check internal Conv2D layer was called with inputs
        self.mock_conv2d_internal_instance.assert_called_once_with(mock_input_tensor)
        
        # 2. Check tf.pad was called with the output of Conv2D and correct padding
        expected_paddings_arg = (
            (0,0), # Batch
            padded_layer.pad, # Height (dilation_rate, dilation_rate)
            padded_layer.pad, # Width (dilation_rate, dilation_rate)
            (0,0)  # Channels
        )
        mock_tf_padded.pad.assert_called_once_with(
            mock_conv_output_tensor,
            expected_paddings_arg,
            mode="REFLECT"
        )
        
        self.assertIs(result, mock_padded_final_output)


    def test_get_config(self):
        padded_layer = PaddedConv2D(
            n_kernels=self.n_kernels_val,
            dilation_rate=self.dilation_rate_val,
            activation=self.activation_val,
            name="get_config_padded"
        )
        mock_parent_config = {'name': 'get_config_padded', 'trainable': True}
        mock_tf_padded.keras.layers.Layer.return_value.get_config.return_value = mock_parent_config.copy()

        config = padded_layer.get_config()

        expected_config = {
            **mock_parent_config,
            'n_kernels': self.n_kernels_val,
            'dilation_rate': self.dilation_rate_val,
            'activation': self.activation_val
        }
        self.assertEqual(config, expected_config)

    @patch.object(PaddedConv2D, '__init__', return_value=None)
    def test_from_config(self, mock_init):
        config_data = {
            'name': 'from_config_padded',
            'n_kernels': 16,
            'dilation_rate': 1,
            'activation': None
        }
        mock_init.reset_mock()
        
        instance = PaddedConv2D.from_config(config_data)
        
        mock_init.assert_called_once_with(**config_data)

    # compute_output_shape is not implemented in PaddedConv2D.
    # If it were, a test would look like:
    # def test_compute_output_shape(self):
    #     layer = PaddedConv2D(...)
    #     input_shape_tensor = tf.TensorShape((None, 32, 32, 3))
    #     output_shape = layer.compute_output_shape(input_shape_tensor)
    #     # Assert expected output shape.
    #     # For PaddedConv2D, since padding='valid' is used internally and then tf.pad REFLECT,
    #     # the H, W dimensions of the output of conv_layer are (input_H - (kernel_H-1)*dilation, input_W - (kernel_W-1)*dilation).
    #     # Then tf.pad adds 2*dilation_rate to H and W.
    #     # If kernel_size is (3,3), then conv_output H = input_H - 2*dilation.
    #     # Padded output H = input_H - 2*dilation + 2*dilation = input_H.
    #     # So, expected output shape would be (None, 32, 32, n_kernels).
    #     self.assertEqual(output_shape.as_list(), [None, 32, 32, self.n_kernels_val])


if __name__ == '__main__':
    unittest.main()
