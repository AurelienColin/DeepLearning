import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from src.trainers.image_to_tag_trainers.comparator_trainer import ComparatorTrainer


class ArtCharComparatorTrainer(ComparatorTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "Comparator"),
            pattern=kwargs.pop('pattern', '.tmp/dataset/AI/*/*/*.webp'),
            input_shape=kwargs.pop('input_shape', (64, 64, 3)),
            batch_size=kwargs.pop('batch_size', 16),
            training_steps=kwargs.pop('training_steps', 1024),
            validation_steps=kwargs.pop('validation_steps', 256),
            layer_kernels=kwargs.pop('layer_kernels', (32, 64, 128)),
            epochs=100,
            *args, **kwargs
        )


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/comparator.py
    trainer = ArtCharComparatorTrainer()
    trainer.model_wrapper.model.summary()
    trainer.run()
