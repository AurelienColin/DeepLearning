from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer
from config import DATASET_ROOT


class RatingTagger(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "Tagger"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/tags/*.json'),
            input_shape=kwargs.pop('input_shape', (192, 192, 3)),
            batch_size=kwargs.pop('batch_size', 12),
            layer_kernels=kwargs.pop('layer_kernels', (32, 64, 128, 128)),
            enforced_tag_names=kwargs.pop(
                'enforced_tag_names',
                ('rating:g', 'rating:s', 'rating:q', 'rating:e')
            ),
            *args, **kwargs)


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/rating_tagger.py
    trainer = RatingTagger()
    trainer.run()
