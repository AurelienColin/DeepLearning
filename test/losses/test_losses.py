import pytest # Added for pytest.approx
import typing

import numpy as np
import tensorflow as tf

from src.losses.losses import (
    Loss, # Still needed by TestLossFunctions for Loss.apply_class_weights
    mae,
    cross_entropy,
    one_minus_dice,
    std_difference,
    edge_loss,
    fourth_channel_mae
)

# Helper to create simple tensors
def create_tensor(data: typing.List[typing.List[typing.List[typing.List[float]]]], dtype=tf.float32) -> tf.Tensor:
    return tf.constant(data, dtype=dtype)

# Removed TestLossFunctions class wrapper, tests are now functions

def test_mae_simple() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    y_pred: tf.Tensor = create_tensor([[[[1.5], [2.5]], [[2.5], [3.5]]]])
    expected_loss: float = 0.5
    loss_val: tf.Tensor = mae(y_true, y_pred)
    assert isinstance(loss_val, tf.Tensor)
    assert float(loss_val.numpy()) == pytest.approx(expected_loss, abs=1e-6)

def test_mae_with_class_weights() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0, 0.0], [2.0, 0.0]]]]) # B, H, W, C
    y_pred: tf.Tensor = create_tensor([[[[1.5, 0.5], [2.5, 0.5]]]])
    class_weights: tf.Tensor = tf.constant([0.5, 1.5], dtype=tf.float32)
    expected_loss: float = 0.5
    loss_val: tf.Tensor = mae(y_true, y_pred, class_weights=class_weights)
    assert isinstance(loss_val, tf.Tensor)
    assert float(loss_val.numpy()) == pytest.approx(expected_loss, abs=1e-6)

def test_cross_entropy_simple() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0], [0.0]]]])
    y_pred: tf.Tensor = create_tensor([[[[0.9], [0.1]]]])
    epsilon: float = 1e-7
    loss_val_tf: tf.Tensor = cross_entropy(y_true, y_pred, epsilon=epsilon)
    expected_loss_val_tensor = - ( (1.0 * tf.math.log(0.9 + epsilon)) + (0.0 * tf.math.log(0.1 + epsilon) + (1.0-0.0) * tf.math.log(1.0-0.1+epsilon)) ) / 2.0
    assert isinstance(loss_val_tf, tf.Tensor)
    assert float(loss_val_tf.numpy()) == pytest.approx(float(expected_loss_val_tensor.numpy()), abs=1e-6)

def test_cross_entropy_with_class_weights() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    y_pred: tf.Tensor = create_tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
    class_weights: tf.Tensor = tf.constant([0.5, 1.5], dtype=tf.float32)
    epsilon: float = 1e-7
    loss_val_tf: tf.Tensor = cross_entropy(y_true, y_pred, class_weights=class_weights, epsilon=epsilon)

    y_true_c0 = y_true[..., 0]
    y_true_c1 = y_true[..., 1]
    y_pred_c0 = y_pred[..., 0]
    y_pred_c1 = y_pred[..., 1]

    loss_c0_p1 = y_true_c0 * tf.math.log(y_pred_c0 + epsilon)
    loss_c0_p2 = (1 - y_true_c0) * tf.math.log(1 - y_pred_c0 + epsilon)
    loss_c0_unweighted = tf.reduce_mean(-(loss_c0_p1 + loss_c0_p2))

    loss_c1_p1 = y_true_c1 * tf.math.log(y_pred_c1 + epsilon)
    loss_c1_p2 = (1 - y_true_c1) * tf.math.log(1 - y_pred_c1 + epsilon)
    loss_c1_unweighted = tf.reduce_mean(-(loss_c1_p1 + loss_c1_p2))

    weighted_losses = tf.constant([loss_c0_unweighted.numpy(), loss_c1_unweighted.numpy()]) * class_weights
    expected_loss_val_tensor = tf.reduce_mean(weighted_losses)

    assert isinstance(loss_val_tf, tf.Tensor)
    assert float(loss_val_tf.numpy()) == pytest.approx(float(expected_loss_val_tensor.numpy()), abs=1e-6)

def test_cross_entropy_epsilon() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0]]]])
    y_pred_zero: tf.Tensor = create_tensor([[[[0.0]]]])
    y_pred_one: tf.Tensor = create_tensor([[[[1.0]]]])
    epsilon: float = 1e-7

    loss_val_pred_zero: tf.Tensor = cross_entropy(y_true, y_pred_zero, epsilon=epsilon)
    expected_pred_zero: float = -tf.math.log(epsilon).numpy()
    assert float(loss_val_pred_zero.numpy()) == pytest.approx(expected_pred_zero, abs=1e-5)

    loss_val_pred_one: tf.Tensor = cross_entropy(y_true, y_pred_one, epsilon=epsilon)
    expected_pred_one: float = -tf.math.log(1.0 + epsilon).numpy()
    assert float(loss_val_pred_one.numpy()) == pytest.approx(expected_pred_one, abs=1e-6)

def test_one_minus_dice_simple() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0], [0.0]], [[1.0], [1.0]]]])
    y_pred: tf.Tensor = create_tensor([[[[1.0], [1.0]], [[1.0], [0.0]]]])
    epsilon: float = 1e-7

    # Corrected expected_loss_val calculation based on per-pixel dice then average
    expected_loss_val = 1.0 - ( (1.0 + 1e-7/(1.0+1e-7) + 1.0 + 1e-7/(1.0+1e-7) )/4.0 )

    loss_val_tf: tf.Tensor = one_minus_dice(y_true, y_pred, epsilon=epsilon)
    assert isinstance(loss_val_tf, tf.Tensor)
    assert float(loss_val_tf.numpy()) == pytest.approx(expected_loss_val, abs=1e-6)

def test_one_minus_dice_with_class_weights() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    y_pred: tf.Tensor = create_tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
    class_weights: tf.Tensor = tf.constant([0.5, 1.5], dtype=tf.float32)
    epsilon: float = 1e-7

    loss_val_tf: tf.Tensor = one_minus_dice(y_true, y_pred, class_weights=class_weights, epsilon=epsilon)

    tp_weighted = Loss.apply_class_weights(y_true * y_pred, class_weights)
    sum_weighted = Loss.apply_class_weights(y_true + y_pred, class_weights)

    num = epsilon + 2 * tp_weighted
    den = epsilon + sum_weighted
    channel_dices = num / den
    expected_loss_val_tensor = 1.0 - tf.reduce_mean(channel_dices)

    assert isinstance(loss_val_tf, tf.Tensor)
    assert float(loss_val_tf.numpy()) == pytest.approx(float(expected_loss_val_tensor.numpy()), abs=1e-6)

def test_one_minus_dice_epsilon() -> None:
    y_true: tf.Tensor = create_tensor([[[[0.0], [0.0]]]])
    y_pred: tf.Tensor = create_tensor([[[[0.0], [0.0]]]])
    epsilon: float = 1e-7
    loss_val_tf: tf.Tensor = one_minus_dice(y_true, y_pred, epsilon=epsilon)
    assert float(loss_val_tf.numpy()) == pytest.approx(0.0, abs=1e-6)

    y_true_one: tf.Tensor = create_tensor([[[[1.0]]]])
    y_pred_one: tf.Tensor = create_tensor([[[[1.0]]]])
    loss_val_tf_ones: tf.Tensor = one_minus_dice(y_true_one, y_pred_one, epsilon=epsilon)
    assert float(loss_val_tf_ones.numpy()) == pytest.approx(0.0, abs=1e-6)

def test_std_difference_simple() -> None:
    y_true_std: tf.Tensor = create_tensor([[[[1.0]]], [[[3.0]]]])
    y_pred_std: tf.Tensor = create_tensor([[[[2.0]]], [[[4.0]]]])
    loss_val_tf: tf.Tensor = std_difference(y_true_std, y_pred_std)
    assert float(loss_val_tf.numpy()) == pytest.approx(0.0, abs=1e-6)

    y_pred_std_diff: tf.Tensor = create_tensor([[[[0.0]]], [[[0.0]]]])
    loss_val_tf_diff: tf.Tensor = std_difference(y_true_std, y_pred_std_diff)
    assert float(loss_val_tf_diff.numpy()) == pytest.approx(1.0, abs=1e-6)

def test_edge_loss_simple() -> None:
    y_true_img: tf.Tensor = create_tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    y_pred_img: tf.Tensor = create_tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    loss_val_tf: tf.Tensor = edge_loss(y_true_img, y_pred_img)
    assert float(loss_val_tf.numpy()) == pytest.approx(0.0, abs=1e-5)

    y_true_manual_larger = tf.constant([
        [
            [[0.0],[0.0],[0.0],[0.0],[0.0]],
            [[0.0],[0.0],[0.0],[0.0],[0.0]],
            [[0.0],[0.0],[1.0],[0.0],[0.0]],
            [[0.0],[0.0],[0.0],[0.0],[0.0]],
            [[0.0],[0.0],[0.0],[0.0],[0.0]]
        ]
    ], dtype=tf.float32)
    y_pred_manual_larger = tf.constant([
        [
            [[0.0],[0.0],[0.0],[0.0],[0.0]],
            [[0.0],[0.0],[0.0],[0.0],[0.0]],
            [[0.0],[0.0],[0.0],[0.0],[0.0]],
            [[0.0],[0.0],[0.0],[0.0],[0.0]],
            [[0.0],[0.0],[0.0],[0.0],[0.0]]
        ]
    ], dtype=tf.float32)
    loss_val_manual: tf.Tensor = edge_loss(y_true_manual_larger, y_pred_manual_larger)
    assert float(loss_val_manual.numpy()) > 0.001

    y_true_edge: tf.Tensor = tf.zeros((1, 5, 5, 1), dtype=tf.float32)
    y_pred_edge: tf.Tensor = tf.pad(tf.ones((1,3,3,1), dtype=tf.float32), [[0,0],[1,1],[1,1],[0,0]])
    loss_val_tf_diff: tf.Tensor = edge_loss(y_true_edge, y_pred_edge)
    assert float(loss_val_tf_diff.numpy()) > 0.0


def test_fourth_channel_mae_simple() -> None:
    y_true: tf.Tensor = create_tensor([[[[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]]]])
    y_pred: tf.Tensor = create_tensor([[[[1.0, 2.0, 3.0, 11.0], [4.0, 5.0, 6.0, 22.0]]]])
    expected_loss: float = 1.5
    loss_val_tf: tf.Tensor = fourth_channel_mae(y_true, y_pred)
    assert float(loss_val_tf.numpy()) == pytest.approx(expected_loss, abs=1e-6)

# TestLossClass has been moved to test_loss_class.py
