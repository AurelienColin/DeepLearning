import pytest
import tensorflow as tf
import warnings


def test_tensorflow_gpu_detection():
    gpu_available = tf.config.list_physical_devices('GPU')
    if not gpu_available:
        warnings.warn(UserWarning("No GPU detected. Running on CPU."))
    assert tf.reduce_sum(tf.ones(2)) == 2
