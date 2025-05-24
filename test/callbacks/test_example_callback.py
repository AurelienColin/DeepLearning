import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock TensorFlow (parent class uses it)
mock_tf_ec = MagicMock() # Use a distinct name to avoid conflict if other test files are loaded
sys.modules['tensorflow'] = mock_tf_ec
sys.modules['tensorflow.keras'] = mock_tf_ec.keras
sys.modules['tensorflow.keras.callbacks'] = mock_tf_ec.keras.callbacks

# Mock Rignak (used by ExampleCallback)
mock_rignak_display = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display

# Mock ModelWrapper (passed to parent Callback's __init__)
mock_model_wrapper_module_ec = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_ec

from callbacks.example_callback import ExampleCallback
# Callback is the parent class, ensure its __init__ and __post_init__ are handled (e.g. os.makedirs)

class TestExampleCallback(unittest.TestCase):

    def setUp(self):
        mock_tf_ec.reset_mock()
        mock_rignak_display.Display.reset_mock() # Reset the Display class mock
        mock_model_wrapper_module_ec.ModelWrapper.reset_mock()
        
        self.mock_model_wrapper_instance = MagicMock(spec=mock_model_wrapper_module_ec.ModelWrapper)
        self.output_path_val = "test/example/output"
        self.mock_display_function = MagicMock(name="DisplayFunction")
        self.mock_display_object = MagicMock(spec=mock_rignak_display.Display) # What function() returns
        self.mock_display_function.return_value = self.mock_display_object

    @patch('os.makedirs') # Mocked for parent Callback's __post_init__
    def create_callback(self, mock_os_makedirs_ignored, period=1, keep_all_epochs=True, **kwargs_parent):
        # This helper ensures os.makedirs is always patched for parent's __post_init__
        params = {
            'model_wrapper': self.mock_model_wrapper_instance,
            'output_path': self.output_path_val,
            'function': self.mock_display_function,
            'period': period,
            'keep_all_epochs': keep_all_epochs,
            **kwargs_parent
        }
        return ExampleCallback(**params)

    def test_initialization(self):
        period_val = 5
        keep_all_val = False
        
        # Call create_callback which internally patches os.makedirs
        callback_instance = self.create_callback(period=period_val, keep_all_epochs=keep_all_val)

        # Assertions for ExampleCallback specific attributes
        self.assertIs(callback_instance.function, self.mock_display_function)
        self.assertEqual(callback_instance.period, period_val)
        self.assertEqual(callback_instance.keep_all_epochs, keep_all_val)
        
        # Assertions for parent class attributes (from Callback)
        self.assertIs(callback_instance.model_wrapper, self.mock_model_wrapper_instance)
        self.assertEqual(callback_instance.output_path, self.output_path_val)
        # Default thumbnail_size if not provided
        self.assertEqual(callback_instance.thumbnail_size, (5,5)) 


    @patch('shutil.copyfile')
    def test_save_display_keep_all_epochs_true(self, mock_shutil_copyfile):
        epoch = 3
        # create_callback patches os.makedirs, so we need it as an arg here
        callback_instance = self.create_callback(keep_all_epochs=True) 
        
        expected_main_file = f"{self.output_path_val}/{epoch:04d}.png"
        expected_alias_file = f"{self.output_path_val}.png"

        returned_filenames = callback_instance.save_display(self.mock_display_object, epoch)

        self.mock_display_object.show.assert_called_once_with(export_filename=expected_main_file)
        mock_shutil_copyfile.assert_called_once_with(expected_main_file, expected_alias_file)
        self.assertEqual(returned_filenames, (expected_main_file, expected_alias_file))

    @patch('shutil.copyfile')
    def test_save_display_keep_all_epochs_false(self, mock_shutil_copyfile):
        epoch = 5
        callback_instance = self.create_callback(keep_all_epochs=False)
        
        expected_main_file = f"{self.output_path_val}.png"
        # No other files expected

        returned_filenames = callback_instance.save_display(self.mock_display_object, epoch)

        self.mock_display_object.show.assert_called_once_with(export_filename=expected_main_file)
        mock_shutil_copyfile.assert_not_called() # No copying if not keeping all epochs
        self.assertEqual(returned_filenames, (expected_main_file,))
        

    @patch.object(ExampleCallback, 'save_display') # Mock the instance method
    def test_on_epoch_end_period_match(self, mock_save_display_method):
        epoch = 4
        period_val = 2 # epoch 4 % period 2 == 0, so should run
        callback_instance = self.create_callback(period=period_val)
        
        # Mock logs dict (though not directly used by this specific on_epoch_end)
        logs_dict = {'loss': 0.5}

        callback_instance.on_epoch_end(epoch, logs=logs_dict)

        self.mock_display_function.assert_called_once()
        mock_save_display_method.assert_called_once_with(self.mock_display_object, epoch)

    @patch.object(ExampleCallback, 'save_display')
    def test_on_epoch_end_period_no_match(self, mock_save_display_method):
        epoch = 3
        period_val = 2 # epoch 3 % period 2 != 0, so should NOT run
        callback_instance = self.create_callback(period=period_val)
        logs_dict = {'loss': 0.6}

        callback_instance.on_epoch_end(epoch, logs=logs_dict)

        self.mock_display_function.assert_not_called()
        mock_save_display_method.assert_not_called()

    @patch.object(ExampleCallback, 'on_epoch_end') # Mock the instance method
    def test_on_train_begin_calls_on_epoch_end_for_epoch_0(self, mock_on_epoch_end_method):
        # on_train_begin should call on_epoch_end(0, ...)
        callback_instance = self.create_callback()
        logs_dict_train_begin = {'initial_metric': 0.1} # Example logs

        callback_instance.on_train_begin(logs=logs_dict_train_begin)

        mock_on_epoch_end_method.assert_called_once_with(0, logs=logs_dict_train_begin)
        # Ensure super().on_train_begin was also called by Keras Callback mock
        # This is tricky as the mock_tf.keras.callbacks.Callback is a generic mock
        # We assume Keras handles calling super if the method exists on the parent.
        # For this test, we focus on the direct logic of ExampleCallback.


if __name__ == '__main__':
    unittest.main()
