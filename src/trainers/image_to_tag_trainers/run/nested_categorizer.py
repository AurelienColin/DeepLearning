from src.trainers.image_to_tag_trainers.nested_categorizer_trainer import NestedCategorizerTrainer
import os

class DanbooruNestedCategorizerTrainer(NestedCategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "DanbooruNested"),
            pattern=kwargs.pop('pattern', '.tmp/dataset/hierarchical/data.json'),
            input_shape=kwargs.pop('input_shape', (256, 256, 3)),
            batch_size=kwargs.pop('batch_size', 24),
            training_steps=kwargs.pop('training_steps', 2048),
            validation_steps=kwargs.pop('validation_steps', 512),
            layer_kernels=kwargs.pop('layer_kernels', (64, 128, 256, 512)),
            epochs=100,
            *args, **kwargs
        )


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/nested_categorizer.py
    trainer = DanbooruNestedCategorizerTrainer()
    trainer.model_wrapper.model.summary()
    trainer.run()
