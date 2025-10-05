from dataclasses import dataclass

import tensorflow as tf

from src.models.image_to_image.auto_encoder_wrapper import AutoEncoderWrapper
from rignak.src.lazy_property import LazyProperty

@dataclass
class UnetWrapper(AutoEncoderWrapper):
    pass
