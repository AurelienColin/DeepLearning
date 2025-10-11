import argparse
import glob
import os
import typing
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from rignak.src.logging_utils import logger

from src.generators.base_generators import BatchGenerator
from rignak.src.lazy_property import LazyProperty
from src.modules.layers.scale_layer import ScaleLayer
from src.modules.custom_objects import CUSTOM_OBJECTS


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a KID from a Keras model.")
    parser.add_argument("--layer_name", type=str, help="Name of the layer to extract.", default="lambda")
    parser.add_argument("--model_path", type=str, help="Path to the Keras model file.",
                        default=".tmp/20250115_095140/model.h5")
    parser.add_argument("--pooling_factor", type=int, default=4, help="Factor for reducing the dimensions.")
    parser.add_argument("--image_size", type=int, default=96, help="Size of the model inputs.")
    parser.add_argument("--dataset", type=str, default="~/Documents/E/GochiUsa/*/*.png", help="Path of the input images.")
    return parser.parse_args()


@dataclass
class Processor:
    model_path: str
    layer_name: str
    dataset: str
    pooling_factor: int = 2
    image_size: int = 96
    n_iterations: int = 100
    batch_size: int = 8

    _original_model: typing.Optional[tf.keras.models.Model] = None
    _partial_model: typing.Optional[tf.keras.models.Model] = None
    _encoder_model: typing.Optional[tf.keras.models.Model] = None
    _model: typing.Optional[tf.keras.models.Model] = None

    @property
    def encoder_filename(self) -> str:
        return os.path.splitext(self.model_path)[0] + '.kid.h5'

    @LazyProperty
    def original_model(self) -> tf.keras.models.Model:
        logger("Load original model", indent=1)
        model = tf.keras.models.load_model(self.model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
        logger("Load original model OK", indent=-1)
        return model

    @LazyProperty
    def partial_model(self) -> tf.keras.models.Model:
        logger("Load partial model", indent=1)
        out_layer = self.original_model.get_layer(self.layer_name).output
        partial_model = tf.keras.Model(inputs=self.original_model.input, outputs=out_layer)
        logger("Load partial model OK", indent=-1)
        return partial_model

    @LazyProperty
    def encoder_model(self) -> tf.keras.models.Model:
        logger("Load encoder model", indent=1)
        out_layer = self.partial_model(self.original_model.input)
        if self.pooling_factor > 1:
            average_pooling = tf.keras.layers.AveragePooling2D(pool_size=self.pooling_factor)
            out_layer = average_pooling(out_layer)
        out_layer = tf.keras.layers.Flatten()(out_layer)
        encoder_model = tf.keras.Model(inputs=self.original_model.input, outputs=out_layer)
        logger("Load encoder model OK", indent=-1)
        return encoder_model

    def get_scale(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        logger("Compute scaling factors", indent=1)
        generator = BatchGenerator(glob.glob(self.dataset), batch_size=self.batch_size,
                                   shape=(self.image_size, self.image_size, 3))

        means = np.zeros(self.encoder_model.output_shape[1])
        stds = np.zeros(self.encoder_model.output_shape[1])

        n = (self.n_iterations * self.batch_size)
        for i in range(self.n_iterations):
            inputs, _ = next(generator)
            means += np.sum(self.encoder_model(inputs).numpy(), axis=0)
        means /= n

        for i in range(self.n_iterations):
            inputs, _ = next(generator)
            stds = np.sum(np.abs(self.encoder_model(inputs).numpy() - means), axis=0)
        stds /= n
        logger("Compute scaling factors OK", indent=1)
        return means, stds

    @LazyProperty
    def model(self) -> tf.keras.models.Model:
        logger("Load scaled encoder model", indent=1)
        means, stds = self.get_scale()

        out_layer = self.encoder_model(self.original_model.input)
        out_layer = ScaleLayer(means, stds)(out_layer)
        scaled_encoder_model = tf.keras.Model(inputs=self.original_model.input, outputs=out_layer)
        scaled_encoder_model.save(self.encoder_filename)
        logger("Load scaled encoder OK", indent=-1)
        return scaled_encoder_model

    def run(self) -> None:
        self.model.summary()
        self.model.save(self.encoder_filename)
        logger(f"Saved processed model to {self.encoder_filename}")

    @classmethod
    def static_run(cls) -> None:
        args = get_args()
        processor = cls(args.model_path, args.layer_name, args.dataset, args.pooling_factor, args.image_size)
        processor.run()


if __name__ == "__main__":
    Processor.static_run()
