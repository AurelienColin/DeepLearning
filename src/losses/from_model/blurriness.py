import tensorflow as tf
import tensorflow.keras.backend as K

from src.losses.from_model.base_loss import LossFromModel


class Blurriness(LossFromModel):
    @property
    def model_path(self) -> str:
        return ".tmp/20250115_095140/model.blurry.h5"

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred_blurriness = self.model(y_pred)
        return K.mean(y_pred_blurriness)
