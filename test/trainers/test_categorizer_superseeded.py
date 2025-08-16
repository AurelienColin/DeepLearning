from keras.src.models.functional import Functional
from ML.src.modules.layers.sparse_conv2d import SparseConv2D
from ML.src.modules.layers.padded_conv2d import PaddedConv2D
from ML.src.modules.layers.atrous_conv2d import AtrousConv2D

from src.trainers.trainer import Trainer
import pytest
from src.trainers.image_to_tag_trainers.run.gochiusa_categorizer import GochiusaCategorizerTrainer


def get_categorizer(layer, **kwargs) -> Trainer:
    categorizer = GochiusaCategorizerTrainer(
        pattern='test/minimal_dataset/gochiusa/*/*.png',
        batch_size=2,
        epochs=1,
        training_steps=5,
        validation_steps=5,
        superseeded_conv_layer=layer,
        superseeded_conv_kwargs=kwargs,
    )
    return categorizer


def check_categorizer_model(categorizer: Trainer):
    model = categorizer.model_wrapper.model
    assert isinstance(model, Functional)
    assert model.input_shape[1:] == categorizer.input_shape
    assert model.output_shape[1:] == categorizer.output_shape
    assert len(model.input_shape) == 4
    assert len(model.output_shape) == 2


def test_baseline():
    categorizer = get_categorizer(PaddedConv2D, dilation_rate=1)
    check_categorizer_model(categorizer)


def test_dilated_x3():
    categorizer = get_categorizer(PaddedConv2D, dilation_rate=3)
    check_categorizer_model(categorizer)


def test_atrous_x3():
    categorizer = get_categorizer(AtrousConv2D, n_stride=3)
    check_categorizer_model(categorizer)


def test_sparse79():
    categorizer = get_categorizer(SparseConv2D, kernel_size=7, n_non_zero=9)
    check_categorizer_model(categorizer)
