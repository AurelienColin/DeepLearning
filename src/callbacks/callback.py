import os
import typing
from dataclasses import dataclass

import tensorflow as tf

from src.models.model_wrapper import ModelWrapper
from rignak.lazy_property import LazyProperty

@dataclass
class Callback(tf.keras.callbacks.Callback):
    model_wrapper: ModelWrapper
    output_path: str
    thumbnail_size: typing.Tuple[int, int] = (5, 5)
    _model: typing.Optional[tf.keras.Model] = None

    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)

    @LazyProperty
    def model(self) -> tf.keras.Model:
        return self.model_wrapper.model