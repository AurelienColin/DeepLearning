import typing

from Rignak.lazy_property import LazyProperty

from src.generators.base_generators import PostProcessGenerator
from src.generators.image_to_image.grey_in_color_out_generator import GreyInColorOutGenerator
from src.trainers.image_to_image_trainers.blurry_autoencoder_trainer import BlurryAutoEncoderTrainer


class Colorizer(BlurryAutoEncoderTrainer):
    def __init__(self, *args, **kwargs):
        super(Colorizer, self).__init__(
            name="GochiUsa",
            pattern='E:\\datasets/GochiUsa/*/*.png',
            input_shape=(96, 96, 1),
            _output_shape=(96, 96, 3),
            batch_size=12,
            *args, **kwargs)

    @LazyProperty
    def post_process_generator_classes(self) -> typing.Sequence[typing.Type[PostProcessGenerator]]:
        return GreyInColorOutGenerator, *super().post_process_generator_classes



if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/colorizer_gochiusa.py
    trainer = Colorizer()
    trainer.run()
