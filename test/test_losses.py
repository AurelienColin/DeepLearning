import pytest
import tensorflow as tf
import numpy as np
import typing

from ML.src.losses.losses import (
    Loss,
    mae,
    cross_entropy,
    cross_entropy_positive,
    one_minus_dice,
    std_difference,
    edge_loss,
    fourth_channel_mae
)


@pytest.fixture
def sample_tensors() -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """Fixture providing simple tensors for testing losses."""
    y_true = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.1, 0.9], [0.8, 0.2]], dtype=tf.float32)
    return y_true, y_pred


@pytest.fixture
def sample_4d_tensors() -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """Fixture providing 4D tensors for image-like data."""
    y_true = tf.constant(
        [[[[0.0, 1.0, 2.0, 0.5],
           [1.0, 2.0, 3.0, 0.3]],
          [[2.0, 3.0, 4.0, 0.7],
           [3.0, 4.0, 5.0, 0.2]]]],
        dtype=tf.float32
    )  # shape (1, 2, 2, 4)
    y_pred = y_true + 0.1
    return y_true, y_pred


# Tests for Loss class

def test_loss_default_weights_initialization(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test that default weights are initialized to ones if not provided."""
    y_true, y_pred = sample_tensors

    def dummy_loss(y_true: tf.Tensor, y_pred: tf.Tensor, class_weights, epsilon: float) -> tf.Tensor:
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    loss_sum = Loss(losses=[dummy_loss, dummy_loss])
    assert isinstance(loss_sum.loss_weights, tf.Tensor)
    np.testing.assert_allclose(loss_sum.loss_weights.numpy(), np.ones(2, dtype=np.float32))

    result = loss_sum(y_true, y_pred)
    assert isinstance(result, tf.Tensor)


def test_loss_custom_weights(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test that custom weights are correctly applied."""
    y_true, y_pred = sample_tensors

    def dummy_loss(y_true, y_pred, class_weights, epsilon):
        return tf.constant(2.0)

    loss_sum = Loss(losses=[dummy_loss, dummy_loss], loss_weights=[0.5, 2.0])
    total = loss_sum(y_true, y_pred)

    expected = 2.0 * 0.5 + 2.0 * 2.0
    np.testing.assert_allclose(total.numpy(), expected)


def test_loss_multiple_losses_combined(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test combining multiple distinct loss functions."""
    y_true, y_pred = sample_tensors

    def loss_1(y_true: tf.Tensor, y_pred: tf.Tensor, class_weights, epsilon: float):
        return tf.reduce_mean((y_true - y_pred) ** 2)

    def loss_2(y_true: tf.Tensor, y_pred: tf.Tensor, class_weights, epsilon: float):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    loss_sum = Loss(losses=[loss_1, loss_2])
    total = loss_sum(y_true, y_pred)

    # Manually compute expected values
    mse = tf.reduce_mean((y_true - y_pred) ** 2)
    mae_val = tf.reduce_mean(tf.abs(y_true - y_pred))
    expected = mse + mae_val

    np.testing.assert_allclose(total.numpy(), expected.numpy(), rtol=1e-6)


def test_loss_with_zero_weight(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test that zero weights effectively disable a loss contribution."""
    y_true, y_pred = sample_tensors

    def loss_1(y_true: tf.Tensor, y_pred: tf.Tensor, class_weights, epsilon: float) -> tf.Tensor:
        return tf.constant(10.0)

    def loss_2(y_true: tf.Tensor, y_pred: tf.Tensor, class_weights, epsilon: float) -> tf.Tensor:
        return tf.constant(1.0)

    loss_sum = Loss(losses=[loss_1, loss_2], loss_weights=[0.0, 1.0])
    total = loss_sum(y_true, y_pred)
    np.testing.assert_allclose(total.numpy(), 1.0)


# Tests for individual loss functions

def test_mae_basic(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test MAE loss function."""
    y_true, y_pred = sample_tensors
    loss = mae(y_true, y_pred)

    expected = tf.reduce_mean(tf.abs(y_true - y_pred))
    np.testing.assert_allclose(loss.numpy(), expected.numpy(), rtol=1e-6)


def test_mae_zero_difference() -> None:
    """MAE should return zero when predictions are perfect."""
    y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

    loss = mae(y_true, y_pred)
    np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-6)


def test_cross_entropy(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test cross entropy loss function."""
    y_true, y_pred = sample_tensors
    loss = cross_entropy(y_true, y_pred)

    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()  # scalar
    assert tf.math.is_finite(loss)


def test_cross_entropy_positive(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test positive cross entropy loss function."""
    y_true, y_pred = sample_tensors
    loss = cross_entropy_positive(y_true, y_pred)

    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()  # scalar
    assert tf.math.is_finite(loss)


def test_one_minus_dice(sample_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test Dice loss function."""
    y_true, y_pred = sample_tensors
    loss = one_minus_dice(y_true, y_pred)

    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()  # scalar
    assert 0.0 <= loss.numpy() <= 1.0  # Dice should be between 0 and 1


def test_one_minus_dice_perfect_prediction() -> None:
    """Dice loss should be zero for perfect predictions."""
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    loss = one_minus_dice(y_true, y_pred)
    assert loss.numpy() < 0.01  # Very close to 0


def test_std_difference() -> None:
    """Test standard deviation difference loss."""
    y_true = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)

    loss = std_difference(y_true, y_pred)
    np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-6)


def test_edge_loss(sample_4d_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test edge loss function using Sobel filters."""
    y_true, y_pred = sample_4d_tensors
    loss = edge_loss(y_true, y_pred)

    assert isinstance(loss, tf.Tensor)
    assert tf.math.is_finite(loss)
    assert loss >= 0.0


def test_edge_loss_identical_images() -> None:
    """Edge loss should be zero for identical images."""
    y_true = tf.random.normal((1, 16, 16, 3))
    y_pred = y_true

    loss = edge_loss(y_true, y_pred)
    np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-6)


def test_fourth_channel_mae(sample_4d_tensors: typing.Tuple[tf.Tensor, tf.Tensor]) -> None:
    """Test fourth channel MAE loss."""
    y_true, y_pred = sample_4d_tensors
    loss = fourth_channel_mae(y_true, y_pred)

    assert isinstance(loss, tf.Tensor)
    assert tf.math.is_finite(loss)

    # Manually compute expected
    expected = tf.reduce_mean(tf.abs(y_true[:, :, :, 3] - y_pred[:, :, :, 3]))
    np.testing.assert_allclose(loss.numpy(), expected.numpy(), rtol=1e-6)


def test_losses_return_scalar() -> None:
    """Ensure all loss functions return scalar values."""
    y_true = tf.random.normal((2, 8, 8, 3))
    y_pred = tf.random.normal((2, 8, 8, 3))

    for loss_fn in [mae, cross_entropy, one_minus_dice, std_difference, edge_loss]:
        result = loss_fn(y_true, y_pred, None)
        assert result.shape == (), f"{loss_fn.__name__} did not return a scalar"


def test_losses_numerical_stability() -> None:
    """Test that losses handle extreme values without NaN/Inf."""
    y_true = tf.constant([[0.0, 1.0]], dtype=tf.float32)
    y_pred = tf.constant([[1e-10, 1.0 - 1e-10]], dtype=tf.float32)

    # Test with epsilon to prevent log(0)
    loss = cross_entropy_positive(y_true, y_pred, epsilon=1e-7)
    assert tf.math.is_finite(loss), "Loss produced NaN or Inf"
