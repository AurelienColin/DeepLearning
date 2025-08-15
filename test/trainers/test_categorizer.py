import os.path

import numpy as np
from keras.src.models.functional import Functional

import shutil
import warnings
from src.trainers.trainer import Trainer
import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def categorizer() -> Trainer:
    from src.trainers.image_to_tag_trainers.run.gochiusa_categorizer import GochiusaCategorizerTrainer
    categorizer = GochiusaCategorizerTrainer(
        pattern='test/minimal_dataset/gochiusa/*/*.png',
        batch_size=2,
        epochs=1,
        training_steps=5,
        validation_steps=5,
    )
    return categorizer


def run(trainer: Trainer) -> None:
    trainer.model_wrapper._output_folder = 'test/.tmp/trainer'
    if os.path.exists(trainer.model_wrapper.output_folder):
        shutil.rmtree(trainer.model_wrapper.output_folder)

    try:
        trainer.run()
    except FileNotFoundError as e:
        warnings.warn(UserWarning(f"FileSystemError encountered: {str(e)}"))
    except Exception as e:
        pytest.fail(f"Computation failed with exception: {str(e)}")


def test_categorizer_generator_types(categorizer: Trainer):
    assert isinstance(categorizer.training_generator, tf.data.Dataset)
    assert isinstance(categorizer.test_generator, tf.data.Dataset)


def test_categorizer_inputs(categorizer: Trainer):
    for generator in (categorizer.training_generator, categorizer.test_generator, categorizer.callback_generator):
        inputs, _ = next(iter(generator))
        assert inputs.shape[1:] == categorizer.input_shape
        assert inputs.shape[0] == categorizer.batch_size


def test_categorizer_outputs(categorizer: Trainer):
    for generator in (categorizer.training_generator, categorizer.test_generator, categorizer.callback_generator):
        _, outputs = next(iter(generator))
        assert outputs.shape[1:] == categorizer.output_shape
        assert outputs.shape[0] == categorizer.batch_size


def test_categorizer_model(categorizer: Trainer):
    model = categorizer.model_wrapper.model
    assert isinstance(model, Functional)
    assert model.input_shape[1:] == categorizer.input_shape
    assert model.output_shape[1:] == categorizer.output_shape
    assert len(model.input_shape) == 4
    assert len(model.output_shape) == 2


def test_training_step(categorizer: Trainer):
    inputs, outputs = next(iter(categorizer.training_generator))
    losses = categorizer.model_wrapper.model.train_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_testing_step(categorizer: Trainer):
    inputs, outputs = next(iter(categorizer.test_generator))
    losses = categorizer.model_wrapper.model.test_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_predict_step(categorizer: Trainer):
    inputs, _ = next(iter(categorizer.callback_generator))
    predictions = categorizer.model_wrapper.model.predict_on_batch((inputs,), )
    assert predictions.shape == (len(inputs), *categorizer.output_shape)
    assert all(np.isfinite(np.stack(predictions).flatten()))


