import typing

import tensorflow as tf

from rignak.lazy_property import LazyProperty
from src.modules.custom_objects import CUSTOM_OBJECTS


class LossFromModel(tf.keras.losses.Loss):
    def __init__(self, name: str, input_shape: typing.Sequence[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = name
        self.input_shape: typing.Sequence[int] = input_shape
        self._model: typing.Optional[tf.keras.models.Model] = None
        self._input_layer: typing.Optional[tf.keras.layers.Layer] = None
        self._tracker: typing.Optional[tf.keras.metrics.Metric] = None

    @property
    def model_path(self) -> str:
        raise NotImplementedError

    @LazyProperty
    def model(self) -> tf.keras.models.Model:
        model = tf.keras.models.load_model(self.model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
        model.trainable = False
        return model

    @LazyProperty
    def input_layer(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Input(shape=self.input_shape)

    @LazyProperty
    def tracker(self) -> tf.keras.metrics.Metric:
        return tf.keras.metrics.Mean(name=f"{self.name}_tracker")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        loss = self.call(y_true, y_pred)
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            loss = tf.multiply(loss, sample_weight)
        self.tracker.update_state(loss)

    def result(self) -> float:
        return self.tracker.result()

    def reset_state(self) -> None:
        self.tracker.reset_state()
