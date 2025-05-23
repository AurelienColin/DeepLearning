from src.trainers.image_to_image_trainers.diffusion_trainer import DiffusionTrainer


class GochiusaDiffusion(DiffusionTrainer):
    def __init__(self, *args, **kwargs):
        super(GochiusaDiffusion, self).__init__(
            name="GochiUsa",
            pattern='E:\\datasets/GochiUsa/*/*.png',
            input_shape=(96, 96, 3),
            batch_size=4,
            *args, **kwargs)



if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/gochiusa_diffusion.py
    trainer = GochiusaDiffusion()
    trainer.run()
