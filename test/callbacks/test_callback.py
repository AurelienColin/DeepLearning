import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock TensorFlow before it's imported (Callback inherits from tf.keras.callbacks.Callback)
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.callbacks'] = mock_tf.keras.callbacks

# Mock ModelWrapper (passed to Callback's __init__)
mock_model_wrapper_module = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module

from callbacks.callback import Callback # This will use the mocked tf and ModelWrapper

class TestCallback(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        mock_model_wrapper_module.ModelWrapper.reset_mock() # Reset the mock class itself

    @patch('os.makedirs')
    def test_initialization_and_post_init(self, mock_os_makedirs):
        # Arrange
        mock_mw_instance = MagicMock(spec=mock_model_wrapper_module.ModelWrapper)
        output_path_val = "test/output/path"
        thumbnail_size_val = (10, 10)

        # Act
        callback_instance = Callback(
            model_wrapper=mock_mw_instance,
            output_path=output_path_val,
            thumbnail_size=thumbnail_size_val
        )

        # Assert attributes are set
        self.assertIs(callback_instance.model_wrapper, mock_mw_instance)
        self.assertEqual(callback_instance.output_path, output_path_val)
        self.assertEqual(callback_instance.thumbnail_size, thumbnail_size_val)

        # Assert __post_init__ behavior
        mock_os_makedirs.assert_called_once_with(output_path_val, exist_ok=True)
        
        # Assert it is an instance of the (mocked) tf.keras.callbacks.Callback
        self.assertIsInstance(callback_instance, mock_tf.keras.callbacks.Callback)

    @patch('os.makedirs')
    def test_initialization_default_thumbnail_size(self, mock_os_makedirs):
        # Arrange
        mock_mw_instance = MagicMock(spec=mock_model_wrapper_module.ModelWrapper)
        output_path_val = "another/path"

        # Act
        callback_instance = Callback(
            model_wrapper=mock_mw_instance,
            output_path=output_path_val
            # thumbnail_size uses default
        )

        # Assert
        self.assertEqual(callback_instance.thumbnail_size, (5, 5)) # Default value
        mock_os_makedirs.assert_called_once_with(output_path_val, exist_ok=True)

    # Test any other concrete methods if Callback had them.
    # For now, it only has __init__ and __post_init__.
    # Standard Keras callback methods like on_epoch_end, on_train_begin, etc.,
    # are not implemented in this base class, so no need to test them here.

if __name__ == '__main__':
    unittest.main()
