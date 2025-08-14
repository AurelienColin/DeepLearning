import typing
import pytest
from ML.src.modules.layers.sparse_conv2d import SparseConv2D
import tensorflow as tf

BATCH_SIZE: int = 4
INPUT_SHAPE: typing.Tuple[int, int] = (16, 8)
N_FILTERS: int = 16
KERNEL_SIZE: int = 7
N_NON_ZEROS: int = 9
N_INPUT_CHANNELS: int = 3

INPUT = tf.random.normal((BATCH_SIZE, *INPUT_SHAPE, N_INPUT_CHANNELS))

@pytest.fixture(scope='module')
def sparse_layer() -> SparseConv2D:
    sparse_layer = SparseConv2D(
        filters=N_FILTERS,
        kernel_size=KERNEL_SIZE,
        n_non_zero=N_NON_ZEROS,
        padding='same',
    )
    sparse_layer.build((None, *INPUT_SHAPE, N_INPUT_CHANNELS))
    return sparse_layer

def test_number_of_trainable_weights(sparse_layer: SparseConv2D) -> None:
    kernel = sparse_layer.trainable_weights[0]
    assert kernel.numpy().nonzero()[0].size == N_NON_ZEROS * N_INPUT_CHANNELS * N_FILTERS

def test_output_shape(sparse_layer: SparseConv2D) -> None:
    output = sparse_layer(INPUT)
    assert output.shape == (BATCH_SIZE, *INPUT_SHAPE, N_FILTERS)
