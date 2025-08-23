import typing

from rignak.lazy_property import LazyProperty

from src.generators.base_generators import PostProcessGenerator
from src.generators.image_to_image.overlay_generator import OverlayGenerator
from src.trainers.image_to_image_trainers.highlighter_trainer import HighlighterTrainer
from config import DATASET_ROOT


class OverlaySegmenter(HighlighterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "overlay"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/overlay/*'),
            input_shape=kwargs.pop('input_shape', (128, 128, 3)),
            batch_size=kwargs.pop('batch_size', 4),
            *args, **kwargs)
        self.base_generator = OverlayGenerator

    @LazyProperty
    def post_process_generator_classes(self) -> typing.Sequence[typing.Type[PostProcessGenerator]]:
        return super().post_process_generator_classes[:-1]


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/overlay_segmenter.py
    trainer = OverlaySegmenter()
    trainer.run()
