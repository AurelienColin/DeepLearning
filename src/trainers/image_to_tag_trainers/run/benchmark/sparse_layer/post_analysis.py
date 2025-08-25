from ML.src.benchmark_utils import main

if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/benchmark/sparse_layer/post_analysis.py
    pattern = ".tmp/benchmark/sparse_layer/*/history.csv"
    main(pattern, n_epochs=10)
