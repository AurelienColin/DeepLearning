import os.path

import numpy as np
from keras.src.models.functional import Functional

import shutil
import warnings
from src.trainers.trainer import Trainer
import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def diffuser() -> Trainer:
    from src.trainers.image_to_image_trainers.run.gochiusa_diffusion import GochiusaDiffusion
    diffuser = GochiusaDiffusion(
        pattern='test/minimal_dataset/gochiusa/*/*.png',
        batch_size=2,
        epochs=1,
        training_steps=5,
        validation_steps=5,
    )
    return diffuser


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


def test_diffuser_generator_types(diffuser: Trainer):
    assert isinstance(diffuser.training_generator, tf.data.Dataset)
    assert isinstance(diffuser.test_generator, tf.data.Dataset)


def test_diffuser_inputs(diffuser: Trainer):
    for generator in (diffuser.training_generator, diffuser.test_generator,
                      diffuser.callback_generator):
        inputs, _ = next(iter(generator))
        assert inputs.shape[1:] == diffuser.input_shape
        assert inputs.shape[0] == diffuser.batch_size


def test_diffuser_outputs(diffuser: Trainer):
    for generator in (diffuser.training_generator, diffuser.test_generator,
                      diffuser.callback_generator):
        _, outputs = next(iter(generator))
        assert outputs.shape[1:] == diffuser.output_shape
        assert outputs.shape[0] == diffuser.batch_size


def test_diffuser_model(diffuser: Trainer):
    model = diffuser.model_wrapper.model
    assert isinstance(model, Functional)
    assert model.input_shape[0][1:] == diffuser.input_shape
    assert model.input_shape[1][1:] == (1, 1, 1)
    assert model.output_shape[1:] == diffuser.output_shape

    assert len(diffuser.input_shape) == 3
    assert len(diffuser.output_shape) == 3


def test_complete_process(diffuser: Trainer):
    if not os.path.exists(diffuser.model_wrapper.model.loss.model_path):
        warnings.warn("KID loss not found. Model can't run.")
    else:
        diffuser.run()
        assert True
