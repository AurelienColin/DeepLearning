import pytest # Added for pytest.approx
import typing
from unittest.mock import patch, MagicMock

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from src.losses.kid import KID # KID_IMAGE_SIZE is KID.KID_IMAGE_SIZE

def assert_tensors_almost_equal(t1: tf.Tensor, t2: tf.Tensor, places: int = 6, msg: str = ""):
    assert isinstance(t1, tf.Tensor), f"First argument is not a Tensor: {msg}"
    assert isinstance(t2, tf.Tensor), f"Second argument is not a Tensor: {msg}"
    val1 = t1.numpy()
    val2 = t2.numpy()
    if np.ndim(val1) == 0 and np.ndim(val2) == 0 : # Scalar comparison
         assert val1 == pytest.approx(val2, abs=10**(-places)), msg
    else: # Array comparison
        np.testing.assert_almost_equal(val1, val2, decimal=places, err_msg=msg)


@patch('src.losses.kid.KID.get_default_encoder')
def test_kid_initialization_default_encoder(mock_get_default_encoder: MagicMock) -> None:
    mock_inner_encoder_layer = MagicMock(spec=tf.keras.layers.Layer)
    mock_inner_encoder_layer.name = 'mock_inner_encoder'

    def default_encoder_layer_side_effect(x, training=None):
        if K.is_keras_tensor(x):
            pooled_x = tf.keras.layers.GlobalAveragePooling2D(name=f"{mock_inner_encoder_layer.name}_sym_pool")(x)
            return tf.keras.layers.Dense(32, name=f"{mock_inner_encoder_layer.name}_sym_dense")(pooled_x)
        return tf.random.normal(shape=(tf.shape(x)[0], 32))

    mock_inner_encoder_layer.side_effect = default_encoder_layer_side_effect
    mock_get_default_encoder.return_value = [mock_inner_encoder_layer] # Must be iterable for KID's * unpack

    kid_metric = KID(name="kid_test", input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))

    mock_get_default_encoder.assert_called_once()
    assert isinstance(kid_metric.encoder, tf.keras.Model)
    assert any(l.name == mock_inner_encoder_layer.name for l in kid_metric.encoder.layers if hasattr(l,'name'))
    assert isinstance(kid_metric.kid_tracker, tf.keras.metrics.Mean)
    assert kid_metric.name == "kid_test"

@patch('src.losses.kid.KID.get_default_encoder')
def test_kid_initialization_custom_encoder(mock_get_default_encoder: MagicMock) -> None:
    mock_custom_layer = MagicMock(spec=tf.keras.layers.Layer)
    mock_custom_layer.name = 'mock_custom_layer'

    def custom_layer_side_effect(x, training=None):
        if K.is_keras_tensor(x):
            pooled_x = tf.keras.layers.GlobalAveragePooling2D(name=f"{mock_custom_layer.name}_sym_pool")(x)
            return tf.keras.layers.Dense(32, name=f"{mock_custom_layer.name}_sym_dense")(pooled_x)
        return tf.random.normal(shape=(tf.shape(x)[0], 32))
    mock_custom_layer.side_effect = custom_layer_side_effect
    custom_layers_with_name_and_side_effect = [mock_custom_layer]

    kid_metric = KID(name="kid_custom", input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3), layers=custom_layers_with_name_and_side_effect)

    mock_get_default_encoder.assert_not_called()
    assert isinstance(kid_metric.encoder, tf.keras.Model)
    sequential_layers = kid_metric.encoder.layers
    assert any(l.name == mock_custom_layer.name for l in sequential_layers if hasattr(l,'name'))
    assert isinstance(kid_metric.kid_tracker, tf.keras.metrics.Mean)

def test_polynomial_kernel() -> None:
    mock_poly_layer = MagicMock(spec=tf.keras.layers.Layer)
    mock_poly_layer.name = 'mock_poly_layer_for_init'
    def poly_layer_side_effect(x, training=None):
        if K.is_keras_tensor(x):
            pooled_x = tf.keras.layers.GlobalAveragePooling2D(name=f"{mock_poly_layer.name}_sym_pool")(x)
            return tf.keras.layers.Dense(32, name=f"{mock_poly_layer.name}_sym_dense")(pooled_x)
        return tf.random.normal(shape=(tf.shape(x)[0], 32))
    mock_poly_layer.side_effect = poly_layer_side_effect

    kid_metric_instance = KID(name="kid_poly_test",
                              input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3),
                              layers=[mock_poly_layer])

    features_1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    features_2 = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    expected_kernel = tf.constant([[3.375, 8.0], [15.625, 27.0]], dtype=tf.float32)

    kernel_output = kid_metric_instance.polynomial_kernel(features_1, features_2)
    assert_tensors_almost_equal(kernel_output, expected_kernel, places=3)


@patch('tensorflow.keras.metrics.Mean')
@patch('tensorflow.keras.models.Sequential')
def test_update_state_calls_encoder_and_tracker(MockSequentialClass: MagicMock, mock_mean_metric: MagicMock) -> None:
    mock_encoder_instance = MockSequentialClass.return_value
    mock_features_concrete = tf.random.normal((10, 32))
    mock_encoder_instance.side_effect = [mock_features_concrete, mock_features_concrete]

    mock_kid_tracker_instance = mock_mean_metric.return_value

    with patch('src.losses.kid.KID.get_default_encoder') as mock_get_default_encoder_init:
        mock_init_layer = MagicMock(spec=tf.keras.layers.Layer); mock_init_layer.name = "init_layer"
        mock_get_default_encoder_init.return_value = [mock_init_layer]
        kid_metric = KID(name="kid_update_test", input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))

    kid_metric.kid_tracker = mock_kid_tracker_instance

    real_images = tf.random.normal((10, KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))
    generated_images = tf.random.normal((10, KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))

    kid_metric.update_state(real_images, generated_images)

    assert mock_encoder_instance.call_count == 2
    assert isinstance(mock_encoder_instance.call_args_list[0][0][0], tf.Tensor)
    assert mock_encoder_instance.call_args_list[0][0][0].shape == real_images.shape
    assert not mock_encoder_instance.call_args_list[0][1]['training']

    assert isinstance(mock_encoder_instance.call_args_list[1][0][0], tf.Tensor)
    assert mock_encoder_instance.call_args_list[1][0][0].shape == generated_images.shape
    assert not mock_encoder_instance.call_args_list[1][1]['training']

    mock_kid_tracker_instance.update_state.assert_called_once()
    assert isinstance(mock_kid_tracker_instance.update_state.call_args[0][0], tf.Tensor)


@patch('tensorflow.keras.metrics.Mean')
@patch('tensorflow.keras.models.Sequential')
def test_update_state_kid_value_calculation(MockSequentialClass: MagicMock, mock_mean_metric: MagicMock) -> None:
    batch_size = 2
    # feature_dim = 4 # Unused

    mock_encoder_instance = MockSequentialClass.return_value
    real_feats_val = tf.constant([[1.,0.,1.,0.], [0.,1.,0.,1.]], dtype=tf.float32)
    gen_feats_val = tf.constant([[1.,1.,0.,0.], [0.,0.,1.,1.]], dtype=tf.float32)
    mock_encoder_instance.side_effect = [real_feats_val, gen_feats_val]

    mock_kid_tracker_instance = mock_mean_metric.return_value
    with patch('src.losses.kid.KID.get_default_encoder') as mock_get_default_encoder_init:
        mock_init_layer = MagicMock(spec=tf.keras.layers.Layer); mock_init_layer.name = "init_layer_val"
        mock_get_default_encoder_init.return_value = [mock_init_layer]
        kid_metric = KID(name="kid_value_calc", input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))

    kid_metric.kid_tracker = mock_kid_tracker_instance

    real_images = tf.zeros((batch_size, KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))
    generated_images = tf.zeros((batch_size, KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))

    kid_metric.update_state(real_images, generated_images)

    k_real = kid_metric.polynomial_kernel(real_feats_val, real_feats_val)
    k_gen = kid_metric.polynomial_kernel(gen_feats_val, gen_feats_val)
    k_cross = kid_metric.polynomial_kernel(real_feats_val, gen_feats_val)

    batch_size_f = tf.cast(batch_size, dtype=tf.float32)
    eye = (1.0 - tf.eye(batch_size))
    norm = batch_size_f * (batch_size_f - 1.0)
    if norm == 0:
        norm = tf.constant(1e-7, dtype=tf.float32)

    mean_k_real = tf.reduce_sum(k_real * eye) / norm
    mean_k_gen = tf.reduce_sum(k_gen * eye) / norm
    mean_k_cross = tf.reduce_mean(k_cross)

    expected_kid_val = mean_k_real + mean_k_gen - 2.0 * mean_k_cross

    mock_kid_tracker_instance.update_state.assert_called_once()
    call_arg = mock_kid_tracker_instance.update_state.call_args[0][0]
    assert_tensors_almost_equal(call_arg, expected_kid_val, places=5)


@patch('tensorflow.keras.metrics.Mean')
@patch('tensorflow.keras.models.Sequential')
def test_result(MockSequentialClass: MagicMock, mock_mean_metric: MagicMock) -> None:
    MockSequentialClass.return_value

    mock_kid_tracker_instance = mock_mean_metric.return_value
    mock_kid_tracker_instance.result.return_value = tf.constant(0.123, dtype=tf.float32)

    with patch('src.losses.kid.KID.get_default_encoder') as mock_get_default_encoder_init:
        mock_init_layer = MagicMock(spec=tf.keras.layers.Layer); mock_init_layer.name = "init_layer_res"
        mock_get_default_encoder_init.return_value = [mock_init_layer]
        kid_metric = KID(name="kid_result_test", input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))
    kid_metric.kid_tracker = mock_kid_tracker_instance

    result = kid_metric.result()
    mock_kid_tracker_instance.result.assert_called_once()
    assert_tensors_almost_equal(result, tf.constant(0.123, dtype=tf.float32))

@patch('tensorflow.keras.metrics.Mean')
@patch('tensorflow.keras.models.Sequential')
def test_reset_state(MockSequentialClass: MagicMock, mock_mean_metric: MagicMock) -> None:
    MockSequentialClass.return_value

    mock_kid_tracker_instance = mock_mean_metric.return_value

    with patch('src.losses.kid.KID.get_default_encoder') as mock_get_default_encoder_init:
        mock_init_layer = MagicMock(spec=tf.keras.layers.Layer); mock_init_layer.name = "init_layer_rst"
        mock_get_default_encoder_init.return_value = [mock_init_layer]
        kid_metric = KID(name="kid_reset_test", input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3))
    kid_metric.kid_tracker = mock_kid_tracker_instance

    kid_metric.reset_state()
    mock_kid_tracker_instance.reset_state.assert_called_once()
