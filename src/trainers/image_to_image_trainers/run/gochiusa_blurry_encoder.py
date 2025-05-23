from src.trainers.image_to_image_trainers.blurry_autoencoder_trainer import BlurryAutoEncoderTrainer


class GochiusaBlurryEncoderTrainer(BlurryAutoEncoderTrainer):
    def __init__(
            self,
            name="GochiUsa",
            pattern='E:\\datasets/GochiUsa/*/*.png',
            input_shape=(96, 96, 3),
            batch_size=12,
            *args,
            **kwargs
    ):
        super(GochiusaBlurryEncoderTrainer, self).__init__(
            name=name, pattern=pattern, input_shape=input_shape, batch_size=batch_size,
            *args, **kwargs
        )



if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/gochiusa_blurry_encoder.py
    trainer = GochiusaBlurryEncoderTrainer()
    trainer.run()
