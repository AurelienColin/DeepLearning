from src.trainers.image_to_tag_trainers.run.gochiusa_categorizer import GochiusaCategorizerTrainer


def main(n_iterations: int = 5):
    for i in range(n_iterations):
        k = 0
        for j, conv in enumerate((tf.keras.Conv2D, atrous_conv2d, sparse_conv2d, padded_conf2d)):
            ...


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/benchmark/background_segmenter/train.py
    main()
