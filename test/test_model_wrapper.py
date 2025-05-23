import unittest
from unittest.mock import MagicMock, patch
import tensorflow as tf

# Adjust the path to import ModelWrapper from src
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.model_wrapper import ModelWrapper

class TestModelWrapper(unittest.TestCase):

    def test_model_wrapper_initialization(self):
        mock_model = MagicMock(spec=tf.keras.models.Model)
        mock_input_shape = (128, 128, 3)
        mock_output_shape = (128, 128, 1)

        wrapper = ModelWrapper(
            model=mock_model,
            input_shape=mock_input_shape,
            output_shape=mock_output_shape,
            name="test_wrapper"
        )

        self.assertEqual(wrapper.model, mock_model)
        self.assertEqual(wrapper.input_shape, mock_input_shape)
        self.assertEqual(wrapper.output_shape, mock_output_shape)
        self.assertEqual(wrapper.name, "test_wrapper")
        self.assertIsNone(wrapper.optimizer)
        self.assertIsNone(wrapper.loss)
        self.assertIsNone(wrapper.metrics)

    def test_compile_method(self):
        mock_model = MagicMock(spec=tf.keras.models.Model)
        wrapper = ModelWrapper(model=mock_model)

        mock_optimizer = MagicMock(spec=tf.keras.optimizers.Optimizer)
        mock_loss = 'mse'
        mock_metrics = ['mae']

        wrapper.compile(
            optimizer=mock_optimizer,
            loss=mock_loss,
            metrics=mock_metrics
        )

        self.assertEqual(wrapper.optimizer, mock_optimizer)
        self.assertEqual(wrapper.loss, mock_loss)
        self.assertEqual(wrapper.metrics, mock_metrics)
        mock_model.compile.assert_called_once_with(
            optimizer=mock_optimizer,
            loss=mock_loss,
            metrics=mock_metrics
        )

    def test_fit_method(self):
        mock_model = MagicMock(spec=tf.keras.models.Model)
        wrapper = ModelWrapper(model=mock_model)

        mock_train_generator = MagicMock()
        mock_val_generator = MagicMock()
        mock_callbacks = [MagicMock()]
        epochs = 10
        initial_epoch = 0

        # Ensure compile has been called, as model.fit would expect a compiled model
        wrapper.compile(optimizer='adam', loss='mse')
        mock_model.reset_mock() # Reset compile mock to focus on fit

        history = wrapper.fit(
            train_generator=mock_train_generator,
            epochs=epochs,
            callbacks=mock_callbacks,
            validation_generator=mock_val_generator,
            initial_epoch=initial_epoch
        )

        mock_model.fit.assert_called_once_with(
            mock_train_generator,
            epochs=epochs,
            callbacks=mock_callbacks,
            validation_data=mock_val_generator,
            initial_epoch=initial_epoch
        )
        self.assertEqual(history, mock_model.fit.return_value)

    def test_fit_method_no_validation(self):
        mock_model = MagicMock(spec=tf.keras.models.Model)
        wrapper = ModelWrapper(model=mock_model)
        mock_train_generator = MagicMock()
        wrapper.compile(optimizer='adam', loss='mse')
        mock_model.reset_mock()

        wrapper.fit(train_generator=mock_train_generator, epochs=5)

        mock_model.fit.assert_called_once_with(
            mock_train_generator,
            epochs=5,
            callbacks=None, # Expecting None if not provided
            validation_data=None, # Expecting None if not provided
            initial_epoch=0 # Expecting default if not provided
        )

    def test_save_and_load_methods(self):
        # These methods directly call keras model's save/load_weights.
        # We mostly test that the call is passed through.
        mock_model = MagicMock(spec=tf.keras.models.Model)
        wrapper = ModelWrapper(model=mock_model)
        filepath = "dummy/path/model.keras"
        custom_objects = {"CustomLayer": MagicMock()}

        wrapper.save(filepath)
        mock_model.save.assert_called_once_with(filepath)

        wrapper.load_weights(filepath)
        mock_model.load_weights.assert_called_once_with(filepath)

        # Test get_model (though simple, for completeness)
        self.assertEqual(wrapper.get_model(), mock_model)

        # Test __str__
        wrapper.name = "MyModel"
        self.assertEqual(str(wrapper), "MyModel")


if __name__ == '__main__':
    unittest.main()
