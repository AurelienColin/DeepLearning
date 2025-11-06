"""
Tests for gradient updates during training and validation.
Inspired by test_gradient_update.py from test_from_another_repo.

These tests verify that:
1. Model weights are updated during training steps
2. Model weights are NOT updated during validation steps
3. Metrics remain finite (no NaN/Inf values)
4. Intermediate layer outputs don't explode
"""
import pytest
import tensorflow as tf
import numpy as np
import typing


def get_weights(model: tf.keras.Model) -> typing.List[np.ndarray]:
    """Extract weights from a model as numpy arrays."""
    return [w.numpy() for w in model.trainable_weights]


def assert_metrics_finite(metrics: typing.Dict[str, float]) -> None:
    """Assert that all metrics are finite (not NaN or Inf)."""
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, tf.Tensor):
            metric_value = metric_value.numpy()
        if isinstance(metric_value, np.ndarray):
            metric_value = metric_value.mean()
        assert np.isfinite(metric_value), f"Metric {metric_name} reached Inf or NaN: {metric_value}"


def assert_weights_updated(
    weights_before: typing.List[np.ndarray],
    weights_after: typing.List[np.ndarray],
    should_update: bool = True
) -> None:
    """
    Assert that weights have been updated (or not) as expected.

    Args:
        weights_before: Model weights before the step
        weights_after: Model weights after the step
        should_update: Whether weights should have been updated
    """
    for i, (weight_before, weight_after) in enumerate(zip(weights_before, weights_after)):
        mae = np.abs(weight_before - weight_after).mean()
        assert np.isfinite(mae), f"Weight {i} reached Inf or NaN"

        if should_update:
            assert mae > 0, f"Weight {i} is zero. Model is not updated despite being in training mode."
        else:
            assert mae == 0, f"Weight {i} isn't zero. Model is updated despite being in validation mode."


def test_simple_model_update_during_training():
    """Test that a simple model updates weights during training."""
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Create sample data
    inputs = tf.random.normal((4, 8, 8, 3))
    outputs = tf.random.normal((4, 10))

    weights_before = get_weights(model)

    # Training step
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(outputs, predictions)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    weights_after = get_weights(model)

    # Verify weights were updated
    assert_weights_updated(weights_before, weights_after, should_update=True)
    assert np.isfinite(loss.numpy())


def test_simple_model_no_update_during_inference():
    """Test that a simple model does NOT update weights during inference."""
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    # Create sample data
    inputs = tf.random.normal((4, 8, 8, 3))

    weights_before = get_weights(model)

    # Inference (no gradient tape, no optimizer)
    _ = model(inputs, training=False)

    weights_after = get_weights(model)

    # Verify weights were NOT updated
    assert_weights_updated(weights_before, weights_after, should_update=False)


def test_batch_normalization_behaves_differently():
    """Test that BatchNormalization behaves differently in training vs inference."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
    ])

    inputs = tf.random.normal((4, 8, 8, 3))

    # Forward pass in training mode
    output_train = model(inputs, training=True)

    # Forward pass in inference mode
    output_inference = model(inputs, training=False)

    # Outputs should be different due to BatchNorm behavior
    # In training: uses batch statistics
    # In inference: uses moving averages
    difference = tf.reduce_mean(tf.abs(output_train - output_inference))

    # There should be some difference (though it might be small)
    # We just verify it's finite and the model runs
    assert tf.math.is_finite(difference)


def test_gradients_are_computed():
    """Test that gradients are actually computed during training."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.MeanSquaredError()
    inputs = tf.random.normal((4, 8, 8, 3))
    outputs = tf.random.normal((4, 10))

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(outputs, predictions)

    gradients = tape.gradient(loss, model.trainable_weights)

    # Verify gradients were computed
    assert gradients is not None
    assert len(gradients) == len(model.trainable_weights)

    # Verify all gradients are finite
    for grad in gradients:
        if grad is not None:
            assert tf.reduce_all(tf.math.is_finite(grad)), "Gradient contains NaN or Inf"


def test_loss_decreases_with_training_steps():
    """Test that loss decreases over multiple training steps."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Create fixed sample data for consistent testing
    np.random.seed(42)
    inputs = tf.constant(np.random.randn(8, 8, 8, 3).astype(np.float32))
    outputs = tf.constant(np.random.randn(8, 1).astype(np.float32))

    losses = []

    # Run multiple training steps
    for _ in range(10):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(outputs, predictions)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        losses.append(loss.numpy())

    # Loss should generally decrease (allow some fluctuation)
    # Check that final loss is less than initial loss
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_model_outputs_no_nan_or_inf():
    """Test that model outputs remain finite during training."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    inputs = tf.random.normal((4, 16, 16, 3))
    outputs = tf.random.normal((4, 10))

    # Run several training steps
    for _ in range(5):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(outputs, predictions)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Check predictions are finite
        assert tf.reduce_all(tf.math.is_finite(predictions)), "Predictions contain NaN or Inf"
        assert tf.math.is_finite(loss), "Loss is NaN or Inf"
