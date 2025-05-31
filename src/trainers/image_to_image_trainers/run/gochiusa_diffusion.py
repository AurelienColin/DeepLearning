from src.trainers.image_to_image_trainers.diffusion_trainer import DiffusionTrainer
from config import DATASET_ROOT


class GochiusaDiffusion(DiffusionTrainer):
    def __init__(self, *args, **kwargs):
        super(GochiusaDiffusion, self).__init__(
            name=kwargs.pop('name', "GochiUsa"),
            pattern=kwargs.pop('pattern', DATASET_ROOT + '/GochiUsa/*/*.png'),
            input_shape=kwargs.pop('input_shape', (96, 96, 3)),
            batch_size=kwargs.pop('batch_size', 4),
            *args, **kwargs)


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/gochiusa_diffusion.py
    trainer = GochiusaDiffusion()
    trainer.run()
