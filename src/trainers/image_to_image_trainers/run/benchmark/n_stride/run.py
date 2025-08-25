from src.trainers.image_to_image_trainers.run.gochiusa_blurry_encoder import GochiusaBlurryEncoderTrainer
from src.modules.layers.atrous_conv2d import AtrousConv2D

def main(n_iterations: int = 5):
    for i in range(n_iterations):
        for n_stride in range(1, 5):
            name = f"benchmark/benchmark_n_stride/{n_stride}_"
            trainer = GochiusaBlurryEncoderTrainer(
            name=name,
            epochs=20,
            superseeded_conv_layer=AtrousConv2D,
            superseeded_conv_kwargs=dict(n_stride=n_stride),
             )
            trainer.run()


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/benchmark/n_stride/run.py
    main()
