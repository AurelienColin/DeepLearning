import unittest
from unittest.mock import MagicMock, patch
import os

# Adjust the path to import Trainer from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trainers.trainer import Trainer
from models.model_wrapper import ModelWrapper
from generators.base_generators import BatchGenerator

class TestTrainer(unittest.TestCase):

    def test_trainer_initialization(self):
        mock_model_wrapper = MagicMock(spec=ModelWrapper)
        mock_train_generator = MagicMock(spec=BatchGenerator)
        mock_val_generator = MagicMock(spec=BatchGenerator)
        mock_callbacks = [MagicMock()]

        trainer = Trainer(
            model_wrapper=mock_model_wrapper,
            train_generator=mock_train_generator,
            val_generator=mock_val_generator,
            epochs=10,
            callbacks=mock_callbacks,
            save_path="dummy/path",
            save_every_N_epochs=5,
            save_best_only=True,
            save_name="test_model",
            starting_epoch=0,
            metrics_on_train_set=True
        )

        self.assertEqual(trainer.model_wrapper, mock_model_wrapper)
        self.assertEqual(trainer.train_generator, mock_train_generator)
        self.assertEqual(trainer.val_generator, mock_val_generator)
        self.assertEqual(trainer.epochs, 10)
        self.assertEqual(trainer.callbacks, mock_callbacks)
        self.assertEqual(trainer.save_path, "dummy/path")
        self.assertEqual(trainer.save_every_N_epochs, 5)
        self.assertTrue(trainer.save_best_only)
        self.assertEqual(trainer.save_name, "test_model")
        self.assertEqual(trainer.starting_epoch, 0)
        self.assertTrue(trainer.metrics_on_train_set)
        self.assertIsNotNone(trainer.history_filename)
        self.assertIsNotNone(trainer.save_filename_template)

    @patch('os.makedirs')
    def test_set_filenames(self, mock_makedirs):
        mock_model_wrapper = MagicMock(spec=ModelWrapper)
        mock_train_generator = MagicMock(spec=BatchGenerator)

        trainer = Trainer(
            model_wrapper=mock_model_wrapper,
            train_generator=mock_train_generator,
            save_path="test/output/models",
            save_name="my_model"
        )

        self.assertEqual(trainer.save_path, "test/output/models")
        self.assertEqual(trainer.save_name, "my_model")
        expected_history_filename = os.path.join("test/output/models", "my_model_history.json")
        self.assertEqual(trainer.history_filename, expected_history_filename)
        expected_save_filename_template = os.path.join("test/output/models", "my_model_epoch_{epoch:04d}.keras")
        self.assertEqual(trainer.save_filename_template, expected_save_filename_template)
        mock_makedirs.assert_called_once_with("test/output/models", exist_ok=True)


    @patch('os.makedirs')
    def test_set_filenames_with_extension(self, mock_makedirs):
        mock_model_wrapper = MagicMock(spec=ModelWrapper)
        mock_train_generator = MagicMock(spec=BatchGenerator)

        trainer = Trainer(
            model_wrapper=mock_model_wrapper,
            train_generator=mock_train_generator,
            save_path="test/output/models",
            save_name="my_model.keras" # Intentionally include extension
        )
        self.assertEqual(trainer.save_name, "my_model") # Extension should be stripped
        expected_history_filename = os.path.join("test/output/models", "my_model_history.json")
        self.assertEqual(trainer.history_filename, expected_history_filename)
        expected_save_filename_template = os.path.join("test/output/models", "my_model_epoch_{epoch:04d}.keras")
        self.assertEqual(trainer.save_filename_template, expected_save_filename_template)
        mock_makedirs.assert_called_once_with("test/output/models", exist_ok=True)

    @patch('os.makedirs')
    def test_set_filenames_no_save_path(self, mock_makedirs):
        mock_model_wrapper = MagicMock(spec=ModelWrapper)
        mock_train_generator = MagicMock(spec=BatchGenerator)

        trainer = Trainer(
            model_wrapper=mock_model_wrapper,
            train_generator=mock_train_generator,
            save_path=None, # No save path
            save_name="my_model"
        )
        self.assertIsNone(trainer.save_path)
        self.assertEqual(trainer.save_name, "my_model")
        self.assertIsNone(trainer.history_filename)
        self.assertIsNone(trainer.save_filename_template)
        mock_makedirs.assert_not_called()


    @patch('tensorflow.keras.callbacks.ModelCheckpoint')
    @patch('tensorflow.keras.callbacks.CSVLogger')
    @patch('src.callbacks.history_callback.HistoryCallback.load_history')
    @patch('src.callbacks.history_callback.HistoryCallback.save_history')
    @patch('os.makedirs')
    def test_run_fit_called_correctly(self, mock_makedirs, mock_save_history, mock_load_history, mock_csv_logger, mock_model_checkpoint):
        mock_model_wrapper = MagicMock(spec=ModelWrapper)
        mock_train_generator = MagicMock(spec=BatchGenerator)
        mock_val_generator = MagicMock(spec=BatchGenerator)
        mock_other_callback = MagicMock()

        trainer = Trainer(
            model_wrapper=mock_model_wrapper,
            train_generator=mock_train_generator,
            val_generator=mock_val_generator,
            epochs=20,
            callbacks=[mock_other_callback],
            save_path="dummy/save",
            save_every_N_epochs=3,
            save_best_only=False,
            save_name="run_model",
            starting_epoch=2
        )

        mock_history_callback_instance = MagicMock()
        mock_load_history.return_value = {} # Start with empty history
        
        # Mock the HistoryCallback constructor to return our mock instance
        with patch('src.callbacks.history_callback.HistoryCallback', return_value=mock_history_callback_instance):
            trainer.run()

        mock_load_history.assert_called_once_with(trainer.history_filename)
        
        self.assertTrue(any(isinstance(call_arg, MagicMock) for call_arg in mock_model_checkpoint.call_args[1]['filepath'] for call_arg in trainer.callbacks if mock_model_checkpoint.call_args ))

        mock_model_checkpoint.assert_called_once_with(
            filepath=trainer.save_filename_template,
            save_best_only=False,
            save_weights_only=False, # Assuming default or common practice
            monitor='val_loss', # Assuming default or common practice
            verbose=1, # Assuming default or common practice
            save_freq='epoch',
            period=3
        )
        
        # Check if CSVLogger was added and configured (if history_filename is present)
        mock_csv_logger.assert_called_once_with(trainer.history_filename, append=True)


        expected_callbacks = [mock_other_callback, mock_history_callback_instance, mock_csv_logger(), mock_model_checkpoint()]
        
        # Verify that model_wrapper.fit was called
        mock_model_wrapper.fit.assert_called_once()
        call_args = mock_model_wrapper.fit.call_args

        self.assertEqual(call_args[1]['train_generator'], mock_train_generator)
        self.assertEqual(call_args[1]['epochs'], 20)
        self.assertEqual(call_args[1]['validation_generator'], mock_val_generator)
        self.assertEqual(call_args[1]['initial_epoch'], 2)
        
        # Check callbacks passed to fit, order might not be guaranteed by trainer
        passed_callbacks_to_fit = call_args[1]['callbacks']
        self.assertIn(mock_other_callback, passed_callbacks_to_fit)
        self.assertIn(mock_history_callback_instance, passed_callbacks_to_fit)
        self.assertTrue(any(isinstance(cb, type(mock_csv_logger())) for cb in passed_callbacks_to_fit))
        self.assertTrue(any(isinstance(cb, type(mock_model_checkpoint())) for cb in passed_callbacks_to_fit))


        mock_save_history.assert_called_once() # History saved at the end

    @patch('os.makedirs')
    def test_run_no_save_path(self, mock_makedirs):
        mock_model_wrapper = MagicMock(spec=ModelWrapper)
        mock_train_generator = MagicMock(spec=BatchGenerator)

        trainer = Trainer(
            model_wrapper=mock_model_wrapper,
            train_generator=mock_train_generator,
            save_path=None,
            epochs=1
        )
        trainer.run()
        mock_model_wrapper.fit.assert_called_once()
        # Ensure no checkpoint or csv logger callbacks were added if no save_path
        for callback in mock_model_wrapper.fit.call_args[1]['callbacks']:
            self.assertNotIsInstance(callback, MagicMock) # If using actual ModelCheckpoint/CSVLogger class for assertion
            # Or if they are mocked, assert they were not called / added.
            # For this example, assuming they are not MagicMock if not added.

if __name__ == '__main__':
    unittest.main()
