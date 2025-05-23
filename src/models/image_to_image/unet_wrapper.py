from dataclasses import dataclass

import tensorflow as tf

from src.models.image_to_image.auto_encoder_wrapper import AutoEncoderWrapper
from Rignak.lazy_property import LazyProperty

@dataclass
class UnetWrapper(AutoEncoderWrapper):
    @LazyProperty
    def encoded_inherited_layers(self) -> tf.keras.layers.Layer:
        self.set_encoded_layers()
        return self._encoded_inherited_layers
