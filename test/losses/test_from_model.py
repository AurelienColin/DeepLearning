import unittest
import typing
from unittest.mock import patch, MagicMock

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# Assuming base_loss.py is in src.losses.from_model, adjust if necessary
from src.losses.from_model.blurriness import Blurriness
from src.losses.from_model.encoding_similarity import EncodingSimilarity

# Helper to create a simple mock Keras model
def create_mock_keras_model(input_shape: typing.Tuple[int, int, int], operation_name: str = "mock_op"):
    """
    Creates a simple Sequential model with a Lambda layer.
    The Lambda layer multiplies its input by 2.0.
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Lambda(lambda x: x * 2.0, name=operation_name)
    ])

class TestBlurrinessMetric(unittest.TestCase):

    def setUp(self):
        self.input_shape = (64, 64, 3) # H, W, C
        self.batch_input_shape = (None,) + self.input_shape # Batch, H, W, C
        self.batch_size = 10

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_blurriness_initialization(self, mock_load_model: MagicMock) -> None:
        # Configure the mock to return a dummy model, though it's not strictly needed for just init
        mock_load_model.return_value = create_mock_keras_model(self.input_shape, "blur_model_init")
        
        metric = Blurriness(name="test_blur", input_shape=self.input_shape)
        self.assertEqual(metric.name, "test_blur")
        # self.assertEqual(metric.input_shape, self.input_shape) # input_shape is not stored directly on instance by base
        
        # Accessing self.model triggers load_model
        self.assertIsNotNone(metric.model)
        mock_load_model.assert_called_once() # With specific path if needed: with(metric.model_path, ...)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_blurriness_call(self, mock_load_model: MagicMock) -> None:
        mock_model = create_mock_keras_model(self.input_shape, "blur_model_call")
        mock_load_model.return_value = mock_model
        
        metric = Blurriness(name="test_blur_call", input_shape=self.input_shape)
        
        y_pred_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred = tf.constant(y_pred_np)
        y_true_dummy = tf.zeros_like(y_pred) # Not used by Blurriness.call

        # Spy on the mock model's call method
        mock_model.call = MagicMock(wraps=mock_model.call)

        loss_value = metric.call(y_true_dummy, y_pred)

        mock_model.call.assert_called_once()
        # Check that the argument passed to the model was y_pred
        # model.call is complex due to __call__ vs call. Easier to check output.

        expected_model_output = y_pred_np * 2.0
        expected_loss = np.mean(expected_model_output)
        
        self.assertIsInstance(loss_value, tf.Tensor)
        np.testing.assert_almost_equal(loss_value.numpy(), expected_loss, decimal=6)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_blurriness_metric_behavior(self, mock_load_model: MagicMock) -> None:
        mock_model = create_mock_keras_model(self.input_shape, "blur_model_metric")
        mock_load_model.return_value = mock_model
        
        metric = Blurriness(name="test_blur_metric", input_shape=self.input_shape)

        # Call 1
        y_pred1_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred1 = tf.constant(y_pred1_np)
        loss1 = np.mean(y_pred1_np * 2.0)
        metric.update_state(tf.zeros_like(y_pred1), y_pred1)
        np.testing.assert_almost_equal(metric.result().numpy(), loss1, decimal=6)

        # Call 2
        y_pred2_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred2 = tf.constant(y_pred2_np)
        loss2 = np.mean(y_pred2_np * 2.0)
        metric.update_state(tf.zeros_like(y_pred2), y_pred2)
        expected_mean_loss = np.mean([loss1, loss2])
        np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

        # Reset
        metric.reset_state()
        self.assertEqual(metric.result().numpy(), 0.0)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_blurriness_update_state_single_call(self, mock_load_model: MagicMock) -> None:
        mock_model = create_mock_keras_model(self.input_shape, "blur_model_update_single")
        mock_load_model.return_value = mock_model
        metric = Blurriness(name="test_blur_update_single", input_shape=self.input_shape)

        y_pred_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred = tf.constant(y_pred_np)
        expected_loss = np.mean(y_pred_np * 2.0)
        
        metric.update_state(tf.zeros_like(y_pred), y_pred)
        np.testing.assert_almost_equal(metric.result().numpy(), expected_loss, decimal=6)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_blurriness_update_state_multiple_calls_and_result(self, mock_load_model: MagicMock) -> None:
        mock_model = create_mock_keras_model(self.input_shape, "blur_model_update_multi")
        mock_load_model.return_value = mock_model
        metric = Blurriness(name="test_blur_update_multi", input_shape=self.input_shape)

        y_pred1_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred1 = tf.constant(y_pred1_np)
        loss1 = np.mean(y_pred1_np * 2.0)
        metric.update_state(tf.zeros_like(y_pred1), y_pred1)

        y_pred2_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred2 = tf.constant(y_pred2_np)
        loss2 = np.mean(y_pred2_np * 2.0)
        metric.update_state(tf.zeros_like(y_pred2), y_pred2)
        
        expected_mean_loss = np.mean([loss1, loss2])
        np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_blurriness_reset_state(self, mock_load_model: MagicMock) -> None:
        mock_model = create_mock_keras_model(self.input_shape, "blur_model_reset")
        mock_load_model.return_value = mock_model
        metric = Blurriness(name="test_blur_reset", input_shape=self.input_shape)

        y_pred_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred = tf.constant(y_pred_np)
        metric.update_state(tf.zeros_like(y_pred), y_pred) # Update once to have a state
        
        metric.reset_state()
        self.assertEqual(metric.result().numpy(), 0.0)


class TestEncodingSimilarityMetric(unittest.TestCase):

    def setUp(self):
        self.input_shape = (64, 64, 3) 
        self.batch_input_shape = (None,) + self.input_shape
        self.batch_size = 10

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_encoding_similarity_initialization(self, mock_load_model: MagicMock) -> None:
        mock_load_model.return_value = create_mock_keras_model(self.input_shape, "enc_model_init")
        
        metric = EncodingSimilarity(name="test_enc_sim", input_shape=self.input_shape)
        self.assertEqual(metric.name, "test_enc_sim")
        
        self.assertIsNotNone(metric.model) # Triggers load_model
        mock_load_model.assert_called_once()

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_encoding_similarity_call(self, mock_load_model: MagicMock) -> None:
        # Use a fresh mock model instance for this test to track its calls specifically
        mock_model_instance = create_mock_keras_model(self.input_shape, "enc_model_call")
        # Spy on the call method of this specific instance
        mock_model_instance.call = MagicMock(wraps=mock_model_instance.call) # Not Sequential.call, but the model itself
        # The Sequential model's __call__ method eventually calls the layers.
        # It's easier to check the number of times load_model's return_value is called.
        # So, we make the return_value of load_model itself a MagicMock wrapping the actual mock_model
        
        # Let's make the mock_model itself a MagicMock that behaves like our Sequential model
        # This allows asserting calls on the model instance returned by load_model
        mock_model_for_test = MagicMock(spec=tf.keras.Sequential)
        mock_model_for_test.side_effect = lambda x: x * 2.0 # Define its behavior

        mock_load_model.return_value = mock_model_for_test
        
        metric = EncodingSimilarity(name="test_enc_sim_call", input_shape=self.input_shape)
        
        y_true_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_true = tf.constant(y_true_np)
        y_pred = tf.constant(y_pred_np)

        loss_value = metric.call(y_true, y_pred)
        
        self.assertEqual(mock_model_for_test.call_count, 2)
        # Check args for first call (y_true)
        # Keras models are called with training=False by default in call if not specified
        # tf.assert_equal(mock_model_for_test.call_args_list[0][0][0], y_true)
        # tf.assert_equal(mock_model_for_test.call_args_list[1][0][0], y_pred)
        # Comparing Tensors directly in mock calls is tricky. Check shapes or a property.
        self.assertEqual(mock_model_for_test.call_args_list[0][0][0].shape, y_true.shape)
        self.assertEqual(mock_model_for_test.call_args_list[1][0][0].shape, y_pred.shape)


        expected_true_encoding = y_true_np * 2.0
        expected_pred_encoding = y_pred_np * 2.0
        expected_loss = np.mean(np.abs(expected_true_encoding - expected_pred_encoding))
        
        self.assertIsInstance(loss_value, tf.Tensor)
        np.testing.assert_almost_equal(loss_value.numpy(), expected_loss, decimal=6)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_encoding_similarity_metric_behavior(self, mock_load_model: MagicMock) -> None:
        # Use a simple lambda for the model's behavior
        mock_model_for_test = MagicMock(spec=tf.keras.Sequential)
        mock_model_for_test.side_effect = lambda x: x * 2.0 
        mock_load_model.return_value = mock_model_for_test
        
        metric = EncodingSimilarity(name="test_enc_sim_metric", input_shape=self.input_shape)

        # Call 1
        y_true1_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred1_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_true1 = tf.constant(y_true1_np); y_pred1 = tf.constant(y_pred1_np)
        loss1 = np.mean(np.abs(y_true1_np * 2.0 - y_pred1_np * 2.0))
        metric.update_state(y_true1, y_pred1)
        np.testing.assert_almost_equal(metric.result().numpy(), loss1, decimal=6)

        # Call 2
        y_true2_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred2_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_true2 = tf.constant(y_true2_np); y_pred2 = tf.constant(y_pred2_np)
        loss2 = np.mean(np.abs(y_true2_np * 2.0 - y_pred2_np * 2.0))
        metric.update_state(y_true2, y_pred2)
        expected_mean_loss = np.mean([loss1, loss2])
        np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

        # Reset
        metric.reset_state()
        self.assertEqual(metric.result().numpy(), 0.0)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_encoding_similarity_update_state_single_call(self, mock_load_model: MagicMock) -> None:
        mock_model = MagicMock(spec=tf.keras.Sequential)
        mock_model.side_effect = lambda x: x * 2.0
        mock_load_model.return_value = mock_model
        metric = EncodingSimilarity(name="test_enc_sim_update_single", input_shape=self.input_shape)

        y_true_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_true = tf.constant(y_true_np); y_pred = tf.constant(y_pred_np)
        expected_loss = np.mean(np.abs(y_true_np * 2.0 - y_pred_np * 2.0))

        metric.update_state(y_true, y_pred)
        np.testing.assert_almost_equal(metric.result().numpy(), expected_loss, decimal=6)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_encoding_similarity_update_state_multiple_calls_and_result(self, mock_load_model: MagicMock) -> None:
        mock_model = MagicMock(spec=tf.keras.Sequential)
        mock_model.side_effect = lambda x: x * 2.0
        mock_load_model.return_value = mock_model
        metric = EncodingSimilarity(name="test_enc_sim_update_multi", input_shape=self.input_shape)

        y_true1_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred1_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_true1 = tf.constant(y_true1_np); y_pred1 = tf.constant(y_pred1_np)
        loss1 = np.mean(np.abs(y_true1_np * 2.0 - y_pred1_np * 2.0))
        metric.update_state(y_true1, y_pred1)

        y_true2_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred2_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_true2 = tf.constant(y_true2_np); y_pred2 = tf.constant(y_pred2_np)
        loss2 = np.mean(np.abs(y_true2_np * 2.0 - y_pred2_np * 2.0))
        metric.update_state(y_true2, y_pred2)
        
        expected_mean_loss = np.mean([loss1, loss2])
        np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

    @patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
    def test_encoding_similarity_reset_state(self, mock_load_model: MagicMock) -> None:
        mock_model = MagicMock(spec=tf.keras.Sequential)
        mock_model.side_effect = lambda x: x * 2.0
        mock_load_model.return_value = mock_model
        metric = EncodingSimilarity(name="test_enc_sim_reset", input_shape=self.input_shape)

        y_true_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_pred_np = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        y_true = tf.constant(y_true_np); y_pred = tf.constant(y_pred_np)
        metric.update_state(y_true, y_pred) # Update once to have a state
        
        metric.reset_state()
        self.assertEqual(metric.result().numpy(), 0.0)

if __name__ == '__main__':
    unittest.main()
