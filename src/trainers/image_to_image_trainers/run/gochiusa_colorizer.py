import typing
import os

from rignak.src.lazy_property import LazyProperty

from src.generators.base_generators import PostProcessGenerator
from src.generators.image_to_image.grey_in_color_out_generator import GreyInColorOutGenerator
from src.trainers.image_to_image_trainers.blurry_autoencoder_trainer import BlurryAutoEncoderTrainer
from src.config import DATASET_ROOT


class Colorizer(BlurryAutoEncoderTrainer):
    def __init__(self, *args, **kwargs):
        super(Colorizer, self).__init__(
            name=kwargs.pop('name', "GochiUsa"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/GochiUsa/*/*.png'),
            input_shape=kwargs.pop('input_shape', (96, 96, 1)),
            _output_shape=kwargs.pop('output_shape', (96, 96, 3)),
            batch_size=kwargs.pop('batch_size', 12),
            *args, **kwargs)

    @LazyProperty
    def post_process_generator_classes(self) -> typing.Sequence[typing.Type[PostProcessGenerator]]:
        return GreyInColorOutGenerator, *super().post_process_generator_classes


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/gochiusa_colorizer.py
    trainer = Colorizer()
    trainer.run()
