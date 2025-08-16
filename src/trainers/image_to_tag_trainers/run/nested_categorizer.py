from src.trainers.image_to_tag_trainers.nested_categorizer_trainer import NestedCategorizerTrainer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class DanbooruNestedCategorizerTrainer(NestedCategorizerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop('name', "DanbooruNested"),
            pattern=kwargs.pop('pattern', '.tmp/dataset/hierarchical/data.json'),
            input_shape=kwargs.pop('input_shape', (96, 96, 3)),
            batch_size=kwargs.pop('batch_size', 12),
            epochs=1,
            *args, **kwargs
        )


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/nested_categorizer.py
    trainer = DanbooruNestedCategorizerTrainer()
    trainer.model_wrapper.model.summary()
    trainer.run()
