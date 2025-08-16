from src.trainers.image_to_tag_trainers.run.gochiusa_categorizer import GochiusaCategorizerTrainer
from ML.src.modules.layers.sparse_conv2d import SparseConv2D
from ML.src.modules.layers.padded_conv2d import PaddedConv2D
from ML.src.modules.layers.atrous_conv2d import AtrousConv2D
import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")


def main(n_iterations: int = 5):
    configs = [
        dict(layer=PaddedConv2D, name="baseline", dilation_rate=1),
        dict(layer=PaddedConv2D, name="dilated3", dilation_rate=3),
        dict(layer=AtrousConv2D, name="atrous3", n_stride=3),
        dict(layer=SparseConv2D, name="sparse79", kernel_size=7, n_non_zero=9),
    ]

    for j, config in enumerate(configs):
        layer = config.pop('layer')
        name = config.pop('name')
        for i in range(n_iterations):
            trainer = GochiusaCategorizerTrainer(
                name=f"benchmark/sparse_layer/{name}_{i}_",
                superseeded_conv_layer=layer,
                superseeded_conv_kwargs=config,
                epochs=10,
            )
            trainer.run()


if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/benchmark/sparse_layer/run.py
    main()
