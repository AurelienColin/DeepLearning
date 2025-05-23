from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer


class Tagger(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Tagger",
            pattern='E:\\datasets/tags/*.json',
            input_shape=(192, 192, 3),
            layer_kernels=(32, 64, 128, 128),
            batch_size=12,
            *args, **kwargs)


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/tagger.py
    trainer = Tagger()
    trainer.run()
