import os.path

import numpy as np
from keras.src.models.functional import Functional

import shutil
import warnings
from src.trainers.trainer import Trainer
import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def transferer() -> Trainer:
    from src.trainers.image_to_image_trainers.run.transfer import TransferTrainer
    transferer = TransferTrainer(
        pattern='test/minimal_dataset/comic33/*/*.png',
        batch_size=2,
        epochs=1,
        training_steps=5,
        validation_steps=5,
    )
    return transferer


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


def test_transferer_generator_types(transferer: Trainer):
    assert isinstance(transferer.training_generator, tf.data.Dataset)
    assert isinstance(transferer.test_generator, tf.data.Dataset)


def test_transferer_inputs(transferer: Trainer):
    for generator in (transferer.training_generator, transferer.test_generator, transferer.callback_generator):
        inputs, _ = next(iter(generator))
        assert inputs.shape[1:] == transferer.input_shape
        assert inputs.shape[0] == transferer.batch_size


def test_transferer_outputs(transferer: Trainer):
    for generator in (transferer.training_generator, transferer.test_generator, transferer.callback_generator):
        _, outputs = next(iter(generator))
        assert outputs.shape[1:] == transferer.output_shape
        assert outputs.shape[0] == transferer.batch_size


def test_transferer_model(transferer: Trainer):
    model = transferer.model_wrapper.model
    assert isinstance(model, Functional)
    assert model.input_shape[1:] == transferer.input_shape
    assert model.output_shape[1:] == transferer.output_shape
    assert len(model.input_shape) == 4
    assert len(model.input_shape) == 4


def test_training_step(transferer: Trainer):
    inputs, outputs = next(iter(transferer.training_generator))
    losses = transferer.model_wrapper.model.train_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_testing_step(transferer: Trainer):
    inputs, outputs = next(iter(transferer.test_generator))
    losses = transferer.model_wrapper.model.test_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_predict_step(transferer: Trainer):
    inputs, _ = next(iter(transferer.callback_generator))
    predictions = transferer.model_wrapper.model.predict_on_batch((inputs,), )
    assert predictions.shape == (len(inputs), *transferer.output_shape)
    assert all(np.isfinite(np.stack(predictions).flatten()))


