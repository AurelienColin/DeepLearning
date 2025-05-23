from src.trainers.image_to_tag_trainers.categorizer_trainer import CategorizerTrainer


class GochiusaCategorizerTrainer(CategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="GochiUsa",
            pattern='E:\\datasets/GochiUsa/*/*.png',
            input_shape=(96, 96, 3),
            batch_size=12,
            *args, **kwargs
        )



if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/gochiusa_categorizer.py
    trainer = GochiusaCategorizerTrainer()
    trainer.run()
