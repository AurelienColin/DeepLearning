import argparse
import os
from dataclasses import dataclass

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import tensorflow as tf
from rignak.src.logging_utils import logger

from rignak.src.lazy_property import LazyProperty

from src.scripts.make_loss_from_encoder import Processor


@dataclass
class NewProcessor(Processor):
    @property
    def encoder_filename(self) -> str:
        return os.path.splitext(self.model_path)[0] + '.blurry.h5'

    @LazyProperty
    def model(self) -> tf.keras.models.Model:
        logger("Load model", indent=1)
        out_layer = self.original_model(self.original_model.input)
        out_layer = tf.keras.layers.Cropping2D(((0, 0), (3, 0)), data_format="channels_first")(out_layer)
        out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
        scaled_encoder_model = tf.keras.Model(inputs=self.original_model.input, outputs=out_layer)
        scaled_encoder_model.save(self.encoder_filename)
        logger("Load model OK", indent=-1)
        return scaled_encoder_model


if __name__ == "__main__":
    NewProcessor.static_run()
