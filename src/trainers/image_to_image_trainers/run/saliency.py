from src.trainers.image_to_image_trainers.highlighter_trainer import HighlighterTrainer
from src.config import DATASET_ROOT


class SaliencyTrainer(HighlighterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "style_transfer128"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/style_transfer128/*/*.jpg'),
            input_shape=kwargs.pop('input_shape', (128, 96, 3)),
            batch_size=kwargs.pop('batch_size', 4),
            *args, **kwargs)


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/saliency.py
    trainer = SaliencyTrainer()
    trainer.run()
