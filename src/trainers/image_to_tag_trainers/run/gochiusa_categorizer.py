from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer
from config import DATASET_ROOT


class GochiusaCategorizerTrainer(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "GochiUsa"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/GochiUsa/*/*.png'),
            input_shape=kwargs.pop('input_shape', (96, 96, 3)),
            batch_size=kwargs.pop('batch_size', 12),
            *args, **kwargs
        )


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/gochiusa_categorizer.py
    trainer = GochiusaCategorizerTrainer()
    trainer.run()
