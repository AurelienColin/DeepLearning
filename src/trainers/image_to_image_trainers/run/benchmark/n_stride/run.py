from src.trainers.image_to_image_trainers.run.gochiusa_blurry_encoder import GochiusaBlurryEncoderTrainer


def main(n_iterations: int = 5):
    for i in range(n_iterations):
        for n_stride in range(1, 5):
            name = f"benchmark/benchmark_n_stride/{n_stride}_"
            trainer = GochiusaBlurryEncoderTrainer(n_stride=n_stride, name=name, epochs=20)
            trainer.run()


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/benchmark/n_stride/run.py
    main()
