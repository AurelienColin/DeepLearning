import pytest # Added for potential future use (e.g. approx) and consistency
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

@pytest.fixture
def metric_setup_data():
    input_shape = (64, 64, 3) # H, W, C
    # batch_input_shape = (None,) + input_shape # Batch, H, W, C - Not directly used by tests after setup
    batch_size = 10
    return {"input_shape": input_shape, "batch_size": batch_size}


@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_blurriness_initialization(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    # Configure the mock to return a dummy model, though it's not strictly needed for just init
    mock_load_model.return_value = create_mock_keras_model(input_shape, "blur_model_init")

    metric = Blurriness(name="test_blur", input_shape=input_shape)
    assert metric.name == "test_blur"

    # Accessing self.model triggers load_model
    assert metric.model is not None
    mock_load_model.assert_called_once() # With specific path if needed: with(metric.model_path, ...)

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_blurriness_call(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = create_mock_keras_model(input_shape, "blur_model_call")
    mock_load_model.return_value = mock_model

    metric = Blurriness(name="test_blur_call", input_shape=input_shape)

    y_pred_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred = tf.constant(y_pred_np)
    y_true_dummy = tf.zeros_like(y_pred) # Not used by Blurriness.call

    # Spy on the mock model's call method
    mock_model.call = MagicMock(wraps=mock_model.call)

    loss_value = metric.call(y_true_dummy, y_pred)

    mock_model.call.assert_called_once()

    expected_model_output = y_pred_np * 2.0
    expected_loss = np.mean(expected_model_output)

    assert isinstance(loss_value, tf.Tensor)
    np.testing.assert_almost_equal(loss_value.numpy(), expected_loss, decimal=6)

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_blurriness_metric_behavior(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = create_mock_keras_model(input_shape, "blur_model_metric")
    mock_load_model.return_value = mock_model

    metric = Blurriness(name="test_blur_metric", input_shape=input_shape)

    # Call 1
    y_pred1_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred1 = tf.constant(y_pred1_np)
    loss1 = np.mean(y_pred1_np * 2.0)
    metric.update_state(tf.zeros_like(y_pred1), y_pred1)
    np.testing.assert_almost_equal(metric.result().numpy(), loss1, decimal=6)

    # Call 2
    y_pred2_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred2 = tf.constant(y_pred2_np)
    loss2 = np.mean(y_pred2_np * 2.0)
    metric.update_state(tf.zeros_like(y_pred2), y_pred2)
    expected_mean_loss = np.mean([loss1, loss2])
    np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

    # Reset
    metric.reset_state()
    assert metric.result().numpy() == 0.0

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_blurriness_update_state_single_call(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = create_mock_keras_model(input_shape, "blur_model_update_single")
    mock_load_model.return_value = mock_model
    metric = Blurriness(name="test_blur_update_single", input_shape=input_shape)

    y_pred_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred = tf.constant(y_pred_np)
    expected_loss = np.mean(y_pred_np * 2.0)

    metric.update_state(tf.zeros_like(y_pred), y_pred)
    np.testing.assert_almost_equal(metric.result().numpy(), expected_loss, decimal=6)

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_blurriness_update_state_multiple_calls_and_result(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = create_mock_keras_model(input_shape, "blur_model_update_multi")
    mock_load_model.return_value = mock_model
    metric = Blurriness(name="test_blur_update_multi", input_shape=input_shape)

    y_pred1_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred1 = tf.constant(y_pred1_np)
    loss1 = np.mean(y_pred1_np * 2.0)
    metric.update_state(tf.zeros_like(y_pred1), y_pred1)

    y_pred2_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred2 = tf.constant(y_pred2_np)
    loss2 = np.mean(y_pred2_np * 2.0)
    metric.update_state(tf.zeros_like(y_pred2), y_pred2)

    expected_mean_loss = np.mean([loss1, loss2])
    np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_blurriness_reset_state(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = create_mock_keras_model(input_shape, "blur_model_reset")
    mock_load_model.return_value = mock_model
    metric = Blurriness(name="test_blur_reset", input_shape=input_shape)

    y_pred_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred = tf.constant(y_pred_np)
    metric.update_state(tf.zeros_like(y_pred), y_pred) # Update once to have a state

    metric.reset_state()
    assert metric.result().numpy() == 0.0


@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_encoding_similarity_initialization(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    mock_load_model.return_value = create_mock_keras_model(input_shape, "enc_model_init")

    metric = EncodingSimilarity(name="test_enc_sim", input_shape=input_shape)
    assert metric.name == "test_enc_sim"

    assert metric.model is not None # Triggers load_model
    mock_load_model.assert_called_once()

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_encoding_similarity_call(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]

    mock_model_for_test = MagicMock(spec=tf.keras.Sequential)
    mock_model_for_test.side_effect = lambda x: x * 2.0 # Define its behavior

    mock_load_model.return_value = mock_model_for_test

    metric = EncodingSimilarity(name="test_enc_sim_call", input_shape=input_shape)

    y_true_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_true = tf.constant(y_true_np)
    y_pred = tf.constant(y_pred_np)

    loss_value = metric.call(y_true, y_pred)

    assert mock_model_for_test.call_count == 2
    assert mock_model_for_test.call_args_list[0][0][0].shape == y_true.shape
    assert mock_model_for_test.call_args_list[1][0][0].shape == y_pred.shape

    expected_true_encoding = y_true_np * 2.0
    expected_pred_encoding = y_pred_np * 2.0
    expected_loss = np.mean(np.abs(expected_true_encoding - expected_pred_encoding))

    assert isinstance(loss_value, tf.Tensor)
    np.testing.assert_almost_equal(loss_value.numpy(), expected_loss, decimal=6)

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_encoding_similarity_metric_behavior(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model_for_test = MagicMock(spec=tf.keras.Sequential)
    mock_model_for_test.side_effect = lambda x: x * 2.0
    mock_load_model.return_value = mock_model_for_test

    metric = EncodingSimilarity(name="test_enc_sim_metric", input_shape=input_shape)

    # Call 1
    y_true1_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred1_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_true1 = tf.constant(y_true1_np); y_pred1 = tf.constant(y_pred1_np)
    loss1 = np.mean(np.abs(y_true1_np * 2.0 - y_pred1_np * 2.0))
    metric.update_state(y_true1, y_pred1)
    np.testing.assert_almost_equal(metric.result().numpy(), loss1, decimal=6)

    # Call 2
    y_true2_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred2_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_true2 = tf.constant(y_true2_np); y_pred2 = tf.constant(y_pred2_np)
    loss2 = np.mean(np.abs(y_true2_np * 2.0 - y_pred2_np * 2.0))
    metric.update_state(y_true2, y_pred2)
    expected_mean_loss = np.mean([loss1, loss2])
    np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

    # Reset
    metric.reset_state()
    assert metric.result().numpy() == 0.0

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_encoding_similarity_update_state_single_call(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = MagicMock(spec=tf.keras.Sequential)
    mock_model.side_effect = lambda x: x * 2.0
    mock_load_model.return_value = mock_model
    metric = EncodingSimilarity(name="test_enc_sim_update_single", input_shape=input_shape)

    y_true_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_true = tf.constant(y_true_np); y_pred = tf.constant(y_pred_np)
    expected_loss = np.mean(np.abs(y_true_np * 2.0 - y_pred_np * 2.0))

    metric.update_state(y_true, y_pred)
    np.testing.assert_almost_equal(metric.result().numpy(), expected_loss, decimal=6)

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_encoding_similarity_update_state_multiple_calls_and_result(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = MagicMock(spec=tf.keras.Sequential)
    mock_model.side_effect = lambda x: x * 2.0
    mock_load_model.return_value = mock_model
    metric = EncodingSimilarity(name="test_enc_sim_update_multi", input_shape=input_shape)

    y_true1_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred1_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_true1 = tf.constant(y_true1_np); y_pred1 = tf.constant(y_pred1_np)
    loss1 = np.mean(np.abs(y_true1_np * 2.0 - y_pred1_np * 2.0))
    metric.update_state(y_true1, y_pred1)

    y_true2_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred2_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_true2 = tf.constant(y_true2_np); y_pred2 = tf.constant(y_pred2_np)
    loss2 = np.mean(np.abs(y_true2_np * 2.0 - y_pred2_np * 2.0))
    metric.update_state(y_true2, y_pred2)

    expected_mean_loss = np.mean([loss1, loss2])
    np.testing.assert_almost_equal(metric.result().numpy(), expected_mean_loss, decimal=6)

@patch('src.losses.from_model.base_loss.tf.keras.models.load_model')
def test_encoding_similarity_reset_state(mock_load_model: MagicMock, metric_setup_data: dict) -> None:
    input_shape = metric_setup_data["input_shape"]
    batch_size = metric_setup_data["batch_size"]
    mock_model = MagicMock(spec=tf.keras.Sequential)
    mock_model.side_effect = lambda x: x * 2.0
    mock_load_model.return_value = mock_model
    metric = EncodingSimilarity(name="test_enc_sim_reset", input_shape=input_shape)

    y_true_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_pred_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_true = tf.constant(y_true_np); y_pred = tf.constant(y_pred_np)
    metric.update_state(y_true, y_pred) # Update once to have a state

    metric.reset_state()
    assert metric.result().numpy() == 0.0
