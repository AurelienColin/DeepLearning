from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer
from src.config import DATASET_ROOT


class Tagger(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "Tagger"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/tags/*.json'),
            input_shape=kwargs.pop('input_shape', (192, 192, 3)),
            layer_kernels=kwargs.pop('layer_kernels', (32, 64, 128, 128)),
            batch_size=kwargs.pop('batch_size', 12),
            *args, **kwargs)


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/tagger.py
    trainer = Tagger()
    trainer.run()
