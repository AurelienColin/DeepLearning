import os
import typing
from dataclasses import dataclass

import tensorflow as tf

from src.models.model_wrapper import ModelWrapper


@dataclass
class Callback(tf.keras.callbacks.Callback):
    model_wrapper: ModelWrapper
    output_path: str
    thumbnail_size: typing.Tuple[int, int] = (5, 5)

    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)
