import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported
mock_tf_scale = MagicMock()
sys.modules['tensorflow'] = mock_tf_scale
sys.modules['tensorflow.keras'] = mock_tf_scale.keras
sys.modules['tensorflow.keras.layers'] = mock_tf_scale.keras.layers

# Mock numpy (used for np.array, np.float32, np.ndarray type hint)
mock_numpy_scale = MagicMock()
# Allow np.ndarray to be used as a type hint
mock_numpy_scale.ndarray = type(MagicMock(name="ndarray_type_mock_scale")())
mock_numpy_scale.float32 = "mock_float32_type" # Placeholder for dtype
# Mock np.array to return the input, so type conversion can be checked if needed
# or make it return a mock that can be used to track calls.
# For this layer, np.array is mainly for type conversion and ensuring float32.
def mock_np_array_func(data, dtype=None):
    # Simulate returning a "numpy array" that might be a list or a mock
    # If we want to check dtype, we can store it or assert it here.
    # For ScaleLayer, it's immediately used in arithmetic, so the mock tensor path is key.
    if isinstance(data, MagicMock): # If it's already a mock, just return it
        return data
    return list(data) # Convert to list to simulate array-like structure for tolist() in get_config

mock_numpy_scale.array.side_effect = mock_np_array_func

sys.modules['numpy'] = mock_numpy_scale


# Import after mocks
from modules.layers.scale_layer import ScaleLayer

class TestScaleLayer(unittest.TestCase):

    def setUp(self):
        mock_tf_scale.reset_mock()
        mock_numpy_scale.reset_mock()
        mock_numpy_scale.array.side_effect = mock_np_array_func # Re-apply side effect

        self.means_list = [0.5, 0.5, 0.5]
        self.stds_list = [0.2, 0.2, 0.2]
        
        # These will be what np.array(means_list, dtype=np.float32) would produce (conceptually)
        # In our mocked environment, they'll be lists or mocks passed to tf ops.
        # For the call method, these will be used in arithmetic with tensors.
        # Let's assume these are broadcastable with input tensors.
        self.mock_means_arr = MagicMock(name="MeansArrayMock")
        self.mock_stds_arr = MagicMock(name="StdsArrayMock")

        # Configure the mock_numpy_scale.array to return these specific mocks when called
        # with the list versions, to simulate the conversion to array-like structures.
        def array_side_effect_for_init(data, dtype=None):
            if data == self.means_list:
                return self.mock_means_arr
            if data == self.stds_list:
                return self.mock_stds_arr
            return list(data) # Fallback for other array calls if any
        mock_numpy_scale.array.side_effect = array_side_effect_for_init
        
        # For get_config, .tolist() is called on means/stds
        self.mock_means_arr.tolist.return_value = self.means_list
        self.mock_stds_arr.tolist.return_value = self.stds_list


    def test_initialization(self):
        scale_layer = ScaleLayer(means=self.means_list, stds=self.stds_list, name="test_scale")

        # Check that np.array was called to convert lists to "numpy arrays" with float32
        mock_numpy_scale.array.assert_any_call(self.means_list, dtype=mock_numpy_scale.float32)
        mock_numpy_scale.array.assert_any_call(self.stds_list, dtype=mock_numpy_scale.float32)
        
        # Check that the (mocked) arrays are stored
        self.assertIs(scale_layer.means, self.mock_means_arr)
        self.assertIs(scale_layer.stds, self.mock_stds_arr)
        
        # Check parent Layer.__init__ call
        mock_tf_scale.keras.layers.Layer.assert_called_with(scale_layer, name="test_scale")


    def test_call_method(self):
        # Re-apply the side effect for this test as it might have been changed by init
        mock_numpy_scale.array.side_effect = array_side_effect_for_init 
        scale_layer = ScaleLayer(means=self.means_list, stds=self.stds_list)
        
        mock_input_tensor = MagicMock(name="InputTensorForScale")
        
        # Mock the arithmetic operations: (inputs - self.means) / self.stds
        # inputs - self.means
        mock_subtracted_tensor = MagicMock(name="SubtractedTensor")
        mock_input_tensor.__sub__ = MagicMock(return_value=mock_subtracted_tensor)
        
        # result_of_subtraction / self.stds
        mock_final_output_tensor = MagicMock(name="FinalOutputTensor")
        mock_subtracted_tensor.__truediv__ = MagicMock(return_value=mock_final_output_tensor)

        result = scale_layer.call(mock_input_tensor)

        # Check subtraction: inputs - self.means (which is self.mock_means_arr)
        mock_input_tensor.__sub__.assert_called_once_with(self.mock_means_arr)
        
        # Check division: (result of sub) / self.stds (which is self.mock_stds_arr)
        mock_subtracted_tensor.__truediv__.assert_called_once_with(self.mock_stds_arr)
        
        self.assertIs(result, mock_final_output_tensor)

        # The prompt mentioned tf.image.resize for ScaleLayer, but the code does arithmetic.
        # So, we assert that resize was NOT called.
        mock_tf_scale.image.resize.assert_not_called()


    def test_get_config(self):
        # Re-apply the side effect for this test
        mock_numpy_scale.array.side_effect = array_side_effect_for_init
        scale_layer = ScaleLayer(means=self.means_list, stds=self.stds_list, name="get_config_scale")
        
        mock_parent_config = {'name': 'get_config_scale', 'trainable': True}
        mock_tf_scale.keras.layers.Layer.return_value.get_config.return_value = mock_parent_config.copy()

        config = scale_layer.get_config()

        # Check that tolist() was called on the numpy array mocks
        self.mock_means_arr.tolist.assert_called_once()
        self.mock_stds_arr.tolist.assert_called_once()

        expected_config = {
            **mock_parent_config,
            'means': self.means_list, # Should be the tolist() version
            'stds': self.stds_list   # Should be the tolist() version
        }
        self.assertEqual(config, expected_config)


    @patch.object(ScaleLayer, '__init__', return_value=None) # Prevent __init__ during direct call
    def test_from_config(self, mock_init):
        config_data_from_get_config = {
            'name': 'from_config_scale',
            'trainable': True, # Example from parent
            'means': [0.1, 0.2], # list format from get_config
            'stds': [0.3, 0.4]
        }
        
        # Mock np.array calls that will happen inside from_config
        # These convert the lists back to "numpy arrays"
        mock_means_arr_from_config = MagicMock(name="MeansArrayFromConfig")
        mock_stds_arr_from_config = MagicMock(name="StdsArrayFromConfig")
        
        # Reset np.array mock and set a new side_effect for from_config context
        mock_numpy_scale.array.reset_mock()
        def from_config_array_side_effect(data, dtype=None):
            if data == config_data_from_get_config['means']:
                return mock_means_arr_from_config
            if data == config_data_from_get_config['stds']:
                return mock_stds_arr_from_config
            return list(data) # Fallback
        mock_numpy_scale.array.side_effect = from_config_array_side_effect
        mock_init.reset_mock() # Reset the __init__ mock

        # Pop 'means' and 'stds' as the classmethod does, then pass rest to constructor
        expected_constructor_config = config_data_from_get_config.copy()
        # The classmethod pops them and passes them as keyword args
        # So, the final call to cls(...) will be:
        # ScaleLayer(means=mock_means_arr_from_config, stds=mock_stds_arr_from_config, **remaining_config)
        
        instance = ScaleLayer.from_config(config_data_from_get_config.copy()) # Pass a copy as it pops

        # Check np.array calls within from_config
        mock_numpy_scale.array.assert_any_call(config_data_from_get_config['means'], dtype=mock_numpy_scale.float32)
        mock_numpy_scale.array.assert_any_call(config_data_from_get_config['stds'], dtype=mock_numpy_scale.float32)
        
        # Check that __init__ (which is mock_init) was called with the processed means/stds and remaining config
        remaining_config = {'name': 'from_config_scale', 'trainable': True} # After pop
        mock_init.assert_called_once_with(
            means=mock_means_arr_from_config,
            stds=mock_stds_arr_from_config,
            **remaining_config
        )

    # compute_output_shape is not implemented in ScaleLayer.
    # If it were, it would likely return input_shape unchanged.
    # def test_compute_output_shape(self):
    #     scale_layer = ScaleLayer(means=self.means_list, stds=self.stds_list)
    #     mock_input_shape_tensor = mock_tf_scale.TensorShape((None, 32, 32, 3))
    #     output_shape = scale_layer.compute_output_shape(mock_input_shape_tensor)
    #     self.assertEqual(output_shape, mock_input_shape_tensor)


if __name__ == '__main__':
    unittest.main()
