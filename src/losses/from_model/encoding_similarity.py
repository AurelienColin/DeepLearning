import typing
from dataclasses import dataclass

import tensorflow as tf
import tensorflow.keras.backend as K

from src.losses.from_model.base_loss import LossFromModel

class EncodingSimilarity(LossFromModel):

    @property
    def model_path(self) -> str:
        return ".tmp/20250115_095140/model.kid.h5"

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_encoding = self.model(y_true)
        y_pred_encoding = self.model(y_pred)
        return K.mean(K.abs(y_true_encoding - y_pred_encoding))
