from src.trainers.image_to_image_trainers.run.background_segmenter import BackgroundSegmenter


def main(n_iterations: int = 5):
    for i in range(n_iterations):
        k = 0
        for n_stride in (1, 3):
            for desactiate_edge_loss in (True, False):
                k+= 1
                name = f"benchmark/benchmark_background_segmenter/{k}_"
                trainer = BackgroundSegmenter(n_stride=n_stride, name=name, epochs=20)
                if desactiate_edge_loss:
                    trainer.model_wrapper.desactivate_edge_loss()
                trainer.run()


if __name__ == "__main__":
    # python src/trainers/image_to_image_trainers/run/benchmark/background_segmenter/train.py
    main()
