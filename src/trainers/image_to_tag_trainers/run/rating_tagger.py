from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer


class RatingTagger(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Tagger",
            pattern='E:\\datasets/tags/*.json',
            input_shape=(192, 192, 3),
            batch_size=12,
            layer_kernels=(32, 64, 128, 128),
            enforced_tag_names=('rating:g', 'rating:s', 'rating:q', 'rating:e'),
            *args, **kwargs)



if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/rating_tagger.py
    trainer = RatingTagger()
    trainer.run()
