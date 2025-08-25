import typing

from rignak.src.lazy_property import LazyProperty

from src.generators.base_generators import PostProcessGenerator
from src.generators.blurry import BlurryGenerator
from src.models.image_to_image.blurry_auto_encoder_wrapper import BlurryAutoEncoderWrapper
from src.trainers.image_to_image_trainers.autoencoder_trainer import AutoEncoderTrainer


class BlurryAutoEncoderTrainer(AutoEncoderTrainer):

    @property
    def output_shape(self) -> typing.Sequence[int]:
        output_shape = super().output_shape
        return *output_shape[:-1], output_shape[-1] + 1

    @property
    def get_model_wrapper(self) -> typing.Type:
        return BlurryAutoEncoderWrapper

    @LazyProperty
    def post_process_generator_classes(self) -> typing.Sequence[typing.Type[PostProcessGenerator]]:
        return *super().post_process_generator_classes, BlurryGenerator
