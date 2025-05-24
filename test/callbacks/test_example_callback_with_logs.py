import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock TensorFlow (parent class uses it)
mock_tf_ecwl = MagicMock() # Use a distinct name
sys.modules['tensorflow'] = mock_tf_ecwl
sys.modules['tensorflow.keras'] = mock_tf_ecwl.keras
sys.modules['tensorflow.keras.callbacks'] = mock_tf_ecwl.keras.callbacks

# Mock Rignak (used by parent ExampleCallback)
mock_rignak_display_ecwl = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_ecwl

# Mock ModelWrapper (passed to grandparent Callback's __init__)
mock_model_wrapper_module_ecwl = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_ecwl

# Mock pandas (used by ExampleCallbackWithLogs)
mock_pandas_ecwl = MagicMock()
sys.modules['pandas'] = mock_pandas_ecwl

from callbacks.example_callback_with_logs import ExampleCallbackWithLogs
# ExampleCallback is the parent class.

class TestExampleCallbackWithLogs(unittest.TestCase):

    def setUp(self):
        mock_tf_ecwl.reset_mock()
        mock_rignak_display_ecwl.Display.reset_mock()
        mock_model_wrapper_module_ecwl.ModelWrapper.reset_mock()
        mock_pandas_ecwl.DataFrame.reset_mock() # Reset DataFrame mock
        mock_pandas_ecwl.DataFrame.return_value.to_csv.reset_mock() # Reset to_csv mock

        self.mock_model_wrapper_instance = MagicMock(spec=mock_model_wrapper_module_ecwl.ModelWrapper)
        self.output_path_val = "test/ecwl/output"
        
        self.mock_display_object = MagicMock(spec=mock_rignak_display_ecwl.Display)
        self.mock_logs_iterable = [{'metric1': 0.1, 'metric2': 0.2}, {'metric1': 0.15, 'metric2': 0.25}]
        
        self.mock_display_and_logs_function = MagicMock(name="DisplayAndLogsFunction")
        self.mock_display_and_logs_function.return_value = (self.mock_display_object, self.mock_logs_iterable)

    @patch('os.makedirs') # Mocked for grandparent Callback's __post_init__
    def create_callback(self, mock_os_makedirs_ignored, period=1, **kwargs_parent):
        params = {
            'model_wrapper': self.mock_model_wrapper_instance,
            'output_path': self.output_path_val,
            'function': self.mock_display_and_logs_function, # This function returns (Display, logs)
            'period': period,
            # keep_all_epochs is inherited from ExampleCallback, defaults to True
            **kwargs_parent 
        }
        return ExampleCallbackWithLogs(**params)

    def test_initialization(self):
        # Mostly tests that parameters are passed to ExampleCallback's init.
        # The type hint for `function` is different, but runtime behavior is based on usage.
        period_val = 3
        
        callback_instance = self.create_callback(period=period_val)

        self.assertIs(callback_instance.function, self.mock_display_and_logs_function)
        self.assertEqual(callback_instance.period, period_val)
        # from parent ExampleCallback
        self.assertTrue(callback_instance.keep_all_epochs) 
        # from grandparent Callback
        self.assertIs(callback_instance.model_wrapper, self.mock_model_wrapper_instance)
        self.assertEqual(callback_instance.output_path, self.output_path_val)


    @patch.object(ExampleCallbackWithLogs, 'save_display') # Mock method from parent
    def test_on_epoch_end_period_match_saves_logs_to_csv(self, mock_save_display_method):
        epoch = 6
        period_val = 3 # epoch 6 % period 3 == 0, should run
        
        # Mock what save_display would return (it returns a tuple of filenames)
        # Example: ("output/path/0006.png", "output/path.png")
        main_display_filename = f"{self.output_path_val}/{epoch:04d}.png"
        alias_display_filename = f"{self.output_path_val}.png"
        mock_save_display_method.return_value = (main_display_filename, alias_display_filename)

        callback_instance = self.create_callback(period=period_val)
        
        # Mock logs dict for the Keras callback signature (though not directly used here)
        keras_logs_dict = {'keras_loss': 0.1}

        callback_instance.on_epoch_end(epoch, logs=keras_logs_dict)

        # 1. Check function call
        self.mock_display_and_logs_function.assert_called_once()
        
        # 2. Check save_display call (from parent)
        mock_save_display_method.assert_called_once_with(self.mock_display_object, epoch)
        
        # 3. Check pandas DataFrame creation and to_csv call
        mock_pandas_ecwl.DataFrame.assert_called_once_with(self.mock_logs_iterable)
        
        expected_csv_filename = f"{self.output_path_val}/{epoch:04d}.csv" # Based on main_display_filename
        # os.path.splitext(main_display_filename)[0] + ".csv"
        mock_pandas_ecwl.DataFrame.return_value.to_csv.assert_called_once_with(expected_csv_filename)


    @patch.object(ExampleCallbackWithLogs, 'save_display')
    def test_on_epoch_end_period_no_match(self, mock_save_display_method):
        epoch = 5
        period_val = 3 # epoch 5 % period 3 != 0, should NOT run main logic
        callback_instance = self.create_callback(period=period_val)
        
        keras_logs_dict = {'keras_loss': 0.2}
        callback_instance.on_epoch_end(epoch, logs=keras_logs_dict)

        self.mock_display_and_logs_function.assert_not_called()
        mock_save_display_method.assert_not_called()
        mock_pandas_ecwl.DataFrame.assert_not_called()
        mock_pandas_ecwl.DataFrame.return_value.to_csv.assert_not_called()

    # on_train_begin is inherited from ExampleCallback and tested there.
    # It calls on_epoch_end(0), so if period allows, CSV would be saved for epoch 0.
    @patch.object(ExampleCallbackWithLogs, 'save_display')
    def test_on_train_begin_saves_csv_if_period_allows_epoch_0(self, mock_save_display_method):
        # Period = 1, so epoch 0 % 1 == 0, should run
        main_display_filename_epoch0 = f"{self.output_path_val}/{0:04d}.png"
        mock_save_display_method.return_value = (main_display_filename_epoch0,) # Keep all epochs false for simplicity

        callback_instance = self.create_callback(period=1, keep_all_epochs=False)
        
        keras_logs_dict_train_begin = {'initial_metric': 0.9}
        callback_instance.on_train_begin(logs=keras_logs_dict_train_begin)

        self.mock_display_and_logs_function.assert_called_once() # Called for epoch 0
        mock_save_display_method.assert_called_once_with(self.mock_display_object, 0)
        mock_pandas_ecwl.DataFrame.assert_called_once_with(self.mock_logs_iterable)
        expected_csv_filename_epoch0 = f"{self.output_path_val}/{0:04d}.csv"
        mock_pandas_ecwl.DataFrame.return_value.to_csv.assert_called_once_with(expected_csv_filename_epoch0)


if __name__ == '__main__':
    unittest.main()
