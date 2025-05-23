import typing

from Rignak.lazy_property import LazyProperty

from src.generators.base_generators import PostProcessGenerator
from src.generators.image_to_image.in_out_substract_generator import InOutSubstractGenerator
from src.trainers.image_to_image_trainers.unet_trainer import UnetTrainer


class HighlighterTrainer(UnetTrainer):

    @property
    def output_shape(self) -> typing.Sequence[int]:
        output_shape = super().output_shape
        return *output_shape[:-1], 1

    @LazyProperty
    def post_process_generator_classes(self) -> typing.Sequence[typing.Type[PostProcessGenerator]]:
        return *super().post_process_generator_classes, InOutSubstractGenerator
