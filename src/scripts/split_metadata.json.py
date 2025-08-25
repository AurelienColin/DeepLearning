import json
import os.path
import sys
from typing import Dict, Sequence

import numpy as np


def split_json(file_path: str, split_ratio: float = 0.75):
    with open(file_path, 'r') as f:
        data: Dict[str, Sequence[str]] = json.load(f)

    keys = list(data.keys())
    np.random.shuffle(keys)

    index = int(split_ratio * len(keys))
    for prefix, s_ in (("training", np.s_[:index]), ("validation", np.s_[index:])):
        output_filename = f"{os.path.dirname(file_path)}/{prefix}_subset.json"
        subset = {'images/' + key: data[key] for key in keys[s_]}
        print(list(subset.items())[0])

        n_tags = len(set([tag for tags in subset.values() for tag in tags]))
        print(f"{prefix} data saved to {output_filename} ({len(subset)} files, {n_tags} tags)")
        with open(output_filename, 'w') as file:
            json.dump(subset, file)


if __name__ == "__main__":
    filename = sys.argv[1]
    split_json(filename)
    # python src/scripts/split_metadata.json.py ~/Documents/E/datasets/tags/metadata.json
