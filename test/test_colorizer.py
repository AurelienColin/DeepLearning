import os.path

import numpy as np
from keras.src.models.functional import Functional

import shutil
import warnings
from src.trainers.trainer import Trainer
import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def colorizer() -> Trainer:
    from src.trainers.image_to_image_trainers.run.gochiusa_colorizer import Colorizer
    colorizer = Colorizer(
        pattern='test/minimal_dataset/gochiusa/*/*.png',
        batch_size=2,
        epochs=1,
        training_steps=5,
        validation_steps=5,
    )
    return colorizer


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


def test_colorizer_generator_types(colorizer: Trainer):
    assert isinstance(colorizer.training_generator, tf.data.Dataset)
    assert isinstance(colorizer.test_generator, tf.data.Dataset)


def test_colorizer_inputs(colorizer: Trainer):
    for generator in (colorizer.training_generator, colorizer.test_generator, colorizer.callback_generator):
        inputs, _ = next(iter(generator))
        assert inputs.shape[1:] == colorizer.input_shape
        assert inputs.shape[0] == colorizer.batch_size


def test_colorizer_outputs(colorizer: Trainer):
    for generator in (colorizer.training_generator, colorizer.test_generator, colorizer.callback_generator):
        _, outputs = next(iter(generator))
        assert outputs.shape[1:] == colorizer.output_shape
        assert outputs.shape[0] == colorizer.batch_size


def test_colorizer_model(colorizer: Trainer):
    model = colorizer.model_wrapper.model
    assert isinstance(model, Functional)
    assert model.input_shape[1:] == colorizer.input_shape
    assert model.output_shape[1:] == colorizer.output_shape
    assert len(model.input_shape) == 4
    assert len(model.input_shape) == 4


def test_training_step(colorizer: Trainer):
    inputs, outputs = next(iter(colorizer.training_generator))
    losses = colorizer.model_wrapper.model.train_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_testing_step(colorizer: Trainer):
    inputs, outputs = next(iter(colorizer.test_generator))
    losses = colorizer.model_wrapper.model.test_on_batch((inputs,), outputs)
    assert all(np.isfinite(np.stack(losses)))


def test_predict_step(colorizer: Trainer):
    inputs, _ = next(iter(colorizer.callback_generator))
    predictions = colorizer.model_wrapper.model.predict_on_batch((inputs,), )
    assert predictions.shape == (len(inputs), *colorizer.output_shape)
    assert all(np.isfinite(np.stack(predictions).flatten()))


def test_complete_process(colorizer: Trainer):
    colorizer.run()
    assert True
