import os.path

import numpy as np
from keras.src.models.functional import Functional

import shutil
import warnings
from src.trainers.trainer import Trainer
import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def blurry_encoder() -> Trainer:
    from src.trainers.image_to_image_trainers.run.gochiusa_blurry_encoder import GochiusaBlurryEncoderTrainer
    blurry_encoder = GochiusaBlurryEncoderTrainer(
        pattern='test/minimal_dataset/gochiusa/*/*.png',
        batch_size=2,
        epochs=1,
        training_steps=5,
        validation_steps=5,
    )
    return blurry_encoder


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


def test_blurry_encoder_generator_types(blurry_encoder: Trainer):
    assert isinstance(blurry_encoder.training_generator, tf.data.Dataset)
    assert isinstance(blurry_encoder.test_generator, tf.data.Dataset)


def test_blurry_encoder_inputs(blurry_encoder: Trainer):
    for generator in (blurry_encoder.training_generator, blurry_encoder.test_generator,
                      blurry_encoder.callback_generator):
        inputs, _ = next(iter(generator))
        assert inputs.shape[1:] == blurry_encoder.input_shape
        assert inputs.shape[0] == blurry_encoder.batch_size


def test_blurry_encoder_outputs(blurry_encoder: Trainer):
    for generator in (blurry_encoder.training_generator, blurry_encoder.test_generator,
                      blurry_encoder.callback_generator):
        _, outputs = next(iter(generator))
        assert outputs.shape[1:] == blurry_encoder.output_shape
        assert outputs.shape[0] == blurry_encoder.batch_size


def test_blurry_encoder_model(blurry_encoder: Trainer):
    model = blurry_encoder.model_wrapper.model
    assert isinstance(model, Functional)
    assert model.input_shape[1:] == blurry_encoder.input_shape
    assert model.output_shape[1:] == blurry_encoder.output_shape
    assert len(model.input_shape) == 4
    assert len(model.input_shape) == 4


def test_training_step(blurry_encoder: Trainer):
    inputs, outputs = next(iter(blurry_encoder.training_generator))
    losses = blurry_encoder.model_wrapper.model.train_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_testing_step(blurry_encoder: Trainer):
    inputs, outputs = next(iter(blurry_encoder.test_generator))
    losses = blurry_encoder.model_wrapper.model.test_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_predict_step(blurry_encoder: Trainer):
    inputs, _ = next(iter(blurry_encoder.callback_generator))
    predictions = blurry_encoder.model_wrapper.model.predict_on_batch((inputs,), )
    assert predictions.shape == (len(inputs), *blurry_encoder.output_shape)
    assert all(np.isfinite(np.stack(predictions).flatten()))


def test_complete_process(blurry_encoder: Trainer):
    blurry_encoder.run()
    assert True
