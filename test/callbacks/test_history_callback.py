import unittest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock TensorFlow (parent class uses it, and self.model is a Keras model)
mock_tf_hc = MagicMock() # Use a distinct name
sys.modules['tensorflow'] = mock_tf_hc
sys.modules['tensorflow.keras'] = mock_tf_hc.keras
sys.modules['tensorflow.keras.callbacks'] = mock_tf_hc.keras.callbacks

# Mock Rignak (used by HistoryCallback)
mock_rignak_display_hc = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_hc

# Mock ModelWrapper (passed to parent Callback's __init__)
mock_model_wrapper_module_hc = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_hc

# Mock numpy and pandas
mock_numpy_hc = MagicMock()
sys.modules['numpy'] = mock_numpy_hc
mock_pandas_hc = MagicMock()
sys.modules['pandas'] = mock_pandas_hc


from callbacks.history_callback import HistoryCallback
# Callback is the parent class.

class TestHistoryCallback(unittest.TestCase):

    def setUp(self):
        mock_tf_hc.reset_mock()
        mock_rignak_display_hc.Display.reset_mock() # Reset Display class mock
        mock_model_wrapper_module_hc.ModelWrapper.reset_mock()
        mock_numpy_hc.reset_mock()
        mock_pandas_hc.DataFrame.reset_mock()
        mock_pandas_hc.DataFrame.return_value.to_csv.reset_mock()

        self.mock_model_wrapper_instance = MagicMock(spec=mock_model_wrapper_module_hc.ModelWrapper)
        # HistoryCallback uses model_wrapper.output_folder
        self.mock_model_wrapper_instance.output_folder = "test/history/model_output"
        
        self.output_path_val = "test/history_callback_output" # For parent Callback
        self.batch_size_val = 32
        self.training_steps_val = 100

        # Mock the Keras model instance that would be set by tf.keras.Model.fit()
        self.mock_keras_model_attribute = MagicMock(spec=mock_tf_hc.keras.Model)
        

    @patch('os.makedirs') # Mocked for grandparent Callback's __post_init__
    def create_callback(self, mock_os_makedirs_ignored, **kwargs_parent):
        params = {
            'model_wrapper': self.mock_model_wrapper_instance,
            'output_path': self.output_path_val, # For parent Callback
            'batch_size': self.batch_size_val,
            'training_steps': self.training_steps_val,
            **kwargs_parent
        }
        callback_instance = HistoryCallback(**params)
        # Manually set the 'model' attribute, as Keras would during fit
        callback_instance.model = self.mock_keras_model_attribute
        return callback_instance

    def test_initialization(self):
        callback_instance = self.create_callback()

        self.assertEqual(callback_instance.thumbnail_size, (12, 6)) # Overridden
        self.assertEqual(callback_instance.ncols, 2)
        self.assertEqual(callback_instance.x, [])
        self.assertEqual(callback_instance.logs, {})
        self.assertEqual(callback_instance.batch_size, self.batch_size_val)
        self.assertEqual(callback_instance.training_steps, self.training_steps_val)
        
        # from grandparent Callback
        self.assertIs(callback_instance.model_wrapper, self.mock_model_wrapper_instance)
        self.assertEqual(callback_instance.output_path, self.output_path_val)

    def test_on_epoch_end_processing_and_plotting(self):
        epoch = 0 # First epoch
        logs_from_keras = {
            'loss': 0.5, 
            'accuracy': 0.8, 
            'val_loss': 0.6, 
            'val_accuracy': 0.75
        }
        
        # Mock np.array to just return the input list for simplicity in checking plot calls
        mock_numpy_hc.array.side_effect = lambda x: list(x) # Convert list of lists to list of lists
        # Mock np.nan for padding
        mock_numpy_hc.nan = "NaN_Val" # Use a string to easily check if it was used

        callback_instance = self.create_callback()
        
        # --- First call to on_epoch_end (epoch 0) ---
        callback_instance.on_epoch_end(epoch, logs=logs_from_keras.copy())

        # 1. Check self.x update
        expected_x_val_epoch0 = (epoch + 1) * self.batch_size_val * self.training_steps_val / 1000
        self.assertEqual(callback_instance.x, [expected_x_val_epoch0])

        # 2. Check self.logs update
        expected_logs_after_epoch0 = {
            'loss': ([0.5], [0.6]), # ([train_loss], [val_loss])
            'accuracy': ([0.8], [0.75]) # ([train_acc], [val_acc])
        }
        self.assertEqual(callback_instance.logs, expected_logs_after_epoch0)

        # 3. Check model saving (attempt save, then save_weights if TypeError)
        self.mock_keras_model_attribute.save.assert_called_once_with(
            self.mock_model_wrapper_instance.output_folder + "/model.h5", include_optimizer=False
        )
        self.mock_keras_model_attribute.save_weights.assert_not_called()

        # 4. Check Display creation
        # nrows = int(np.ceil(len(self.logs) / self.ncols)) = ceil(2/2) = 1
        mock_numpy_hc.ceil.return_value = 1.0 # Mock ceil to return 1 for nrows=1
        mock_display_instance = mock_rignak_display_hc.Display.return_value
        mock_rignak_display_hc.Display.assert_called_once_with(
            figsize=callback_instance.thumbnail_size, nrows=1, ncols=callback_instance.ncols
        )

        # 5. Check plotting calls
        # For 'loss'
        plot_call_loss = call(
            [expected_x_val_epoch0], # x values
            [[0.5], [0.6]],       # y values (transposed from logs['loss'])
            xlabel='kimgs', yscale="log", title='loss', labels=('loss', 'val_loss')
        )
        # For 'accuracy'
        plot_call_accuracy = call(
            [expected_x_val_epoch0],    # x values
            [[0.8], [0.75]],          # y values (transposed from logs['accuracy'])
            xlabel='kimgs', yscale="log", title='accuracy', labels=('accuracy', 'val_accuracy')
        )
        # mock_display_instance is a MagicMock, its __getitem__ returns another MagicMock
        # which then has a `plot` method.
        # Assuming logs are processed in order: loss then accuracy
        self.assertEqual(mock_display_instance.__getitem__.call_count, 2)
        mock_display_instance.__getitem__.return_value.plot.assert_any_call(*plot_call_loss[1], **plot_call_loss[2])
        mock_display_instance.__getitem__.return_value.plot.assert_any_call(*plot_call_accuracy[1], **plot_call_accuracy[2])
        
        # 6. Check display show
        mock_display_instance.show.assert_called_once_with(
            export_filename=self.mock_model_wrapper_instance.output_folder + "/history.png"
        )

        # 7. Check CSV saving
        expected_df_data_for_csv = {'loss': [0.6], 'accuracy': [0.75]} # Only validation values
        mock_pandas_hc.DataFrame.assert_called_once_with(expected_df_data_for_csv)
        mock_pandas_hc.DataFrame.return_value.to_csv.assert_called_once_with(
            self.mock_model_wrapper_instance.output_folder + "/history.csv"
        )

        # --- Second call to on_epoch_end (epoch 1, logs missing val_accuracy) ---
        epoch = 1
        logs_epoch1_missing_val_acc = {'loss': 0.4, 'accuracy': 0.85, 'val_loss': 0.5}
        self.mock_keras_model_attribute.save.reset_mock() # Reset for second call
        mock_rignak_display_hc.Display.reset_mock()
        mock_display_instance.reset_mock() # Reset instance for new calls
        mock_pandas_hc.DataFrame.reset_mock()
        mock_pandas_hc.DataFrame.return_value.to_csv.reset_mock()
        mock_numpy_hc.ceil.return_value = 1.0 # Still 2 items in self.logs keys

        callback_instance.on_epoch_end(epoch, logs=logs_epoch1_missing_val_acc.copy())
        
        expected_x_val_epoch1 = (epoch + 1) * self.batch_size_val * self.training_steps_val / 1000
        self.assertEqual(callback_instance.x, [expected_x_val_epoch0, expected_x_val_epoch1])
        
        # Check logs update with np.nan padding
        expected_logs_after_epoch1 = {
            'loss': ([0.5, 0.4], [0.6, 0.5]), 
            'accuracy': ([0.8, 0.85], [0.75, "NaN_Val"]) # val_accuracy padded
        }
        self.assertEqual(callback_instance.logs, expected_logs_after_epoch1)
        self.mock_keras_model_attribute.save.assert_called_once() # Called again

        # Check CSV data for epoch 1 (should only contain current epoch's val data)
        # The code is: pd.DataFrame({key: value[1] for key, value in self.logs.items()})
        # This means it takes the *entire history* of validation values for each key.
        expected_df_data_epoch1_csv = {'loss': [0.6, 0.5], 'accuracy': [0.75, "NaN_Val"]}
        mock_pandas_hc.DataFrame.assert_called_once_with(expected_df_data_epoch1_csv)
        mock_pandas_hc.DataFrame.return_value.to_csv.assert_called_once()


    def test_on_epoch_end_model_save_type_error_uses_save_weights(self):
        epoch = 0
        logs_from_keras = {'loss': 0.5}
        callback_instance = self.create_callback()

        # Configure model.save to raise TypeError to trigger save_weights
        self.mock_keras_model_attribute.save.side_effect = TypeError("Mocked TypeError for save")

        callback_instance.on_epoch_end(epoch, logs=logs_from_keras)

        self.mock_keras_model_attribute.save.assert_called_once() # Attempted
        self.mock_keras_model_attribute.save_weights.assert_called_once_with(
            self.mock_model_wrapper_instance.output_folder + "/model.h5", include_optimizer=False
        )

    def test_on_epoch_end_no_logs_from_keras(self):
        epoch = 0
        callback_instance = self.create_callback()
        
        # Mock np.nan for padding
        mock_numpy_hc.nan = "NaN_Val_NoLogs"
        
        # Call with logs=None
        callback_instance.on_epoch_end(epoch, logs=None)
        
        # self.logs should remain empty or be padded with NaNs if keys were added previously.
        # Since it's the first call and logs is None, no keys are added from logs.
        # However, the code iterates logs.items(). If logs is empty, self.logs remains empty.
        # The padding logic `if base_key not in logs:` and `if 'val_' + base_key not in logs:`
        # implies that if a key *was* in self.logs from a previous epoch, but not in current `logs`,
        # it would get NaN padding. But here, no keys are ever added if `logs` is empty.
        self.assertEqual(callback_instance.logs, {})

        # Check Display creation (nrows should be 0 if self.logs is empty)
        mock_numpy_hc.ceil.return_value = 0.0 
        mock_rignak_display_hc.Display.assert_called_once_with(figsize=(12,6), nrows=0, ncols=2)
        
        # No plotting should occur if self.logs is empty
        mock_display_instance = mock_rignak_display_hc.Display.return_value
        mock_display_instance.__getitem__.assert_not_called()
        
        # Display.show and pd.DataFrame(...).to_csv should still be called
        mock_display_instance.show.assert_called_once()
        mock_pandas_hc.DataFrame.assert_called_once_with({}) # Empty dict from self.logs
        mock_pandas_hc.DataFrame.return_value.to_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
