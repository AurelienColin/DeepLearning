import pytest
import warnings

import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_USE_CUDNN_AUTOTUNE"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.random.set_seed(1)

def test_has_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    assert gpus


def test_reduce_sum():
    with tf.device('/GPU:0'):
        assert tf.reduce_sum(tf.ones(2)) == 2


def test_convolution():
    input_shape = (1, 8, 8, 3)
    kernel_shape = (3, 3, 3, 8)

    x = tf.random.normal(input_shape)
    kernel = tf.random.normal(kernel_shape)

    with tf.device('/GPU:0'):
        y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        result = tf.reduce_sum(y).numpy()
    assert result is not None