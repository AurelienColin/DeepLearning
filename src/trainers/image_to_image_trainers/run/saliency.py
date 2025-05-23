from src.trainers.image_to_image_trainers.highlighter_trainer import HighlighterTrainer


class SaliencyTrainer(HighlighterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="uncensor128",
            pattern='E:\\datasets/style_transfer128/*/*.jpg',
            input_shape=(128, 96, 3),
            batch_size=4,
            *args, **kwargs)



if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/saliency.py
    trainer = SaliencyTrainer()
    trainer.run()
