import datetime
import typing
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics
from rignak.lazy_property import LazyProperty
from rignak.logging_utils import logger

from src.on_model_start import write_summary, backup


@dataclass
class ModelWrapper:
    name: str
    input_shape: typing.Sequence[int]
    batch_size: int
    _output_shape: typing.Optional[typing.Sequence[int]] = None
    learning_rate: float = 1e-5
    loss_weights: typing.Sequence[float] = (1.,)

    _output_folder: typing.Optional[str] = None
    training_generator: typing.Optional[typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]] = None
    test_generator: typing.Optional[typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]] = None

    _input_layer: typing.Optional[tf.keras.layers.Layer] = None
    _output_layer: typing.Optional[tf.keras.layers.Layer] = None
    _input_layers: typing.Optional[typing.Sequence[tf.keras.layers.Layer]] = None
    _model: typing.Optional[tf.keras.models.Model] = None

    _optimizer: typing.Optional[tf.keras.optimizers.Optimizer] = None
    _loss: typing.Optional[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None
    _metrics: typing.Optional[typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]] = None
    _metrics_for_keras: typing.Optional[typing.Sequence[tf.keras.metrics.Metric]] = None

    @LazyProperty
    def loss(self) -> typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        raise NotImplementedError

    @LazyProperty
    def metrics(self) -> typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        raise NotImplementedError

    @property
    def output_shape(self) -> typing.Sequence[int]:
        return self.input_shape if self._output_shape is None else self._output_shape

    @LazyProperty
    def output_folder(self) -> str:
        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return f".tmp/{self.name}_{self.__class__.__name__}_{date}"

    @LazyProperty
    def input_layer(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Input(shape=self.input_shape)

    @property
    def input_layers(self) -> typing.Sequence[tf.keras.layers.Layer]:
        return self.input_layer,

    @property
    def output_layer(self) -> typing.Sequence[tf.keras.layers.Layer]:
        raise NotImplementedError

    @LazyProperty
    def model(self) -> tf.keras.models.Model:
        logger(f"Setup model", indent=1)
        model = tf.keras.models.Model(self.input_layers, self.output_layer)
        logger(f"Setup model OK", indent=-1)
        return model

    def on_start(self) -> None:
        self.compile()
        write_summary(self.model, f"{self.output_folder}/model.txt")
        backup(f"{self.output_folder}/backup.zip")

    @LazyProperty
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1000.)

    @LazyProperty
    def metrics_for_keras(self) -> typing.Sequence[tf.keras.metrics.Metric]:
        metrics = [
            tf.keras.metrics.MeanMetricWrapper(fn=metric, name=metric.__name__)
            for metric in self.metrics
            if metric is not None
        ]
        return metrics


    def compile(self) -> None:
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics_for_keras
        )

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
