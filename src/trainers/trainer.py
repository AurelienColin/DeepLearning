import glob
import typing
from dataclasses import dataclass

import matplotlib
import numpy as np
import tensorflow as tf
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger

from src.callbacks.history_callback import HistoryCallback
from src.generators.base_generators import BatchGenerator, PostProcessGenerator, compose_generators
from src.generators.image_to_image.symmetry_generator import VerticalSymmetryGenerator
from src.models.model_wrapper import ModelWrapper

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
matplotlib.use('agg')

tf.config.run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be set per GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error setting memory growth: {e}")


@dataclass
class Trainer:
    name: str
    pattern: str
    input_shape: typing.Tuple[int, int, int]
    batch_size: int
    _output_shape: typing.Optional[typing.Sequence] = None

    training_steps: int = 512
    validation_steps: int = 128
    epochs: int = 100
    n_stride: int = 4

    on_start: bool= True

    layer_kernels: typing.Optional[typing.Sequence[int]] = None
    enforced_tag_names: typing.Optional[typing.Sequence[str]] = None

    _model_wrapper: typing.Optional[ModelWrapper] = None
    _filenames: typing.Optional[typing.Sequence[str]] = None
    _training_filenames: typing.Optional[typing.Sequence[str]] = None
    _validation_filenames: typing.Optional[typing.Sequence[str]] = None
    _training_generator: typing.Optional[BatchGenerator] = None
    _test_generator: typing.Optional[BatchGenerator] = None
    _callback_generator: typing.Optional[BatchGenerator] = None
    _callbacks: typing.Optional[typing.Sequence[tf.keras.callbacks.Callback]] = None

    _post_process_generator_classes: typing.Sequence[typing.Type[PostProcessGenerator]] = None

    _class_weight: typing.Optional[np.ndarray] = None

    _original_training_generator: typing.Optional[BatchGenerator] = None
    base_generator: typing.Optional[typing.Type] = None

    superseeded_conv_layer: typing.Optional[tf.keras.layers.Layer] = None
    superseeded_conv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None

    @property
    def class_weight(self) -> typing.Optional[np.ndarray]:
        return None

    @property
    def output_shape(self) -> typing.Sequence:
        return self.input_shape if self._output_shape is None else self._output_shape

    @LazyProperty
    def model_wrapper(self) -> ModelWrapper:
        self.set_model_wrapper()
        return self._model_wrapper

    @property
    def get_model_wrapper(self) -> typing.Type:
        raise NotImplementedError

    def set_model_wrapper(self) -> None:
        logger(f"Setup model_wrapper", indent=1)
        kwargs = dict(layer_kernels=self.layer_kernels, _output_shape=self.output_shape)
        self._model_wrapper = self.get_model_wrapper(
            self.name,
            self.input_shape,
            self.batch_size,
            training_generator=self.training_generator,
            test_generator=self.test_generator,
            superseeded_conv_layer=self.superseeded_conv_layer,
            superseeded_conv_kwargs=self.superseeded_conv_kwargs,
            **{key: value for key, value in kwargs.items() if value is not None}
        )
        if self.on_start:
            self._model_wrapper.on_start()
        logger(f"Setup model_wrapper OK", indent=-1)

    @staticmethod
    def split_names_on_pattern(pattern: str) -> typing.Tuple[typing.Sequence[str], typing.Sequence[str]]:
        filenames = glob.glob(pattern)
        n_filenames = len(filenames)

        if n_filenames < 2:
            raise FileNotFoundError(f"Pattern `{pattern}` has {n_filenames} matchs.")
        elif n_filenames > 2:
            np.random.shuffle(filenames)
            index = int(n_filenames * 0.75)
            return filenames[:index], filenames[index:]
        else:
            return (filenames[0],), (filenames[1],)

    def set_filenames(self) -> None:
        logger(f"Setup filenames", indent=1)
        self._training_filenames, self._validation_filenames = self.split_names_on_pattern(self.pattern)
        logger(f"Setup filenames OK", indent=-1)

    @LazyProperty
    def training_filenames(self) -> typing.Sequence[str]:
        self.set_filenames()
        return self._training_filenames

    @LazyProperty
    def validation_filenames(self) -> typing.Sequence[str]:
        self.set_filenames()
        return self._validation_filenames

    @LazyProperty
    def callbacks(self) -> typing.Sequence[tf.keras.callbacks.Callback]:
        callbacks = [HistoryCallback(
            model_wrapper=self.model_wrapper,
            output_path=self.model_wrapper.output_folder,
            batch_size=self.batch_size,
            training_steps=self.training_steps,
        )]
        return callbacks

    @LazyProperty
    def post_process_generator_classes(self) -> typing.Sequence[typing.Type]:
        return VerticalSymmetryGenerator,

    @property
    def get_base_generator(self) -> typing.Type:
        if self.base_generator is None:
            raise NotImplementedError
        return self.base_generator

    def get_generator(self, *args, base_generator: typing.Optional[BatchGenerator] = None) -> BatchGenerator:
        if base_generator is None:
            base_generator = self.base_generator(*args, **self.generator_kwargs)
        return compose_generators(base_generator, self.post_process_generator_classes)

    @property
    def generator_kwargs(self) -> typing.Dict:
        kwargs = dict(enforced_tag_names=self.enforced_tag_names)
        return {key: value for key, value in kwargs.items() if value is not None}

    @LazyProperty
    def original_training_generator(self) -> BatchGenerator:
        return self.get_base_generator(
            self.training_filenames,
            self.batch_size,
            self.input_shape,
            **self.generator_kwargs
        )

    @LazyProperty
    def training_generator(self) -> tf.data.Dataset:
        logger(f"Setup training_generator", indent=1)
        generator = self.get_generator(base_generator=self.original_training_generator)
        dataset = self.generator_to_dataset(generator)
        dataset.output_space = generator.output_space
        logger(f"Setup training_generator OK", indent=-1)
        return dataset

    @LazyProperty
    def test_generator(self, **kwargs) -> tf.data.Dataset:
        logger(f"Setup test_generator", indent=1)
        generator = self.get_generator(self.validation_filenames, self.batch_size, self.input_shape)
        dataset = self.generator_to_dataset(generator)
        dataset.output_space = generator.output_space
        logger(f"Setup test_generator OK", indent=-1)
        return dataset

    @LazyProperty
    def callback_generator(self, **kwargs) -> BatchGenerator:
        logger(f"Setup callback_generator", indent=1)
        generator = self.get_generator(self.validation_filenames, self.batch_size, self.input_shape)
        # generator = self.generator_to_dataset(generator)
        logger(f"Setup callback_generator OK", indent=-1)
        return generator

    @staticmethod
    def generator_to_dataset(generator: BatchGenerator) -> tf.data.Dataset:
        sample_x, sample_y = next(generator)
        output_shapes = (tf.TensorShape((None, *sample_x.shape[1:])),
                         tf.TensorShape((None, *sample_y.shape[1:])))
        output_types = (tf.float32, tf.float32)

        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        return dataset

    def run(self):
        logger(f"Run trainer")

        self.model_wrapper.fit(
            self.model_wrapper.training_generator,
            batch_size=self.batch_size,
            validation_data=self.model_wrapper.test_generator,
            steps_per_epoch=self.training_steps,
            validation_steps=self.validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
            class_weight=self.class_weight,
        )
