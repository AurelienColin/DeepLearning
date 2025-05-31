from src.trainers.image_to_image_trainers.blurry_autoencoder_trainer import BlurryAutoEncoderTrainer
from config import DATASET_ROOT


class GochiusaBlurryEncoderTrainer(BlurryAutoEncoderTrainer):
    def __init__(self, *args, **kwargs):
        super(GochiusaBlurryEncoderTrainer, self).__init__(
            name=kwargs.pop('name', "GochiUsa"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/GochiUsa/*/*.png'),
            input_shape=kwargs.pop('input_shape', (96, 96, 3)),
            batch_size=kwargs.pop('batch_size', 12),
            *args, **kwargs
        )


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/gochiusa_blurry_encoder.py
    trainer = GochiusaBlurryEncoderTrainer()
    trainer.run()
