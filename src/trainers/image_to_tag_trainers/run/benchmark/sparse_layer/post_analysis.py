from src.trainers.image_to_tag_trainers.run.benchmark.utils import main

if __name__ == "__main__":
    pattern = ".tmp/benchmark/sparse_layer/*/history.csv"
    main(pattern)
