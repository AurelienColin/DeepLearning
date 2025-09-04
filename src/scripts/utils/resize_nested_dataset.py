import typing

from src.output_spaces.custom.nested.nested_space import NestedSpace
from src.samples.sample import Sample
from rignak.src.logging_utils import logger
import PIL.Image
import os
import sys
import json
import numpy as np


def get_new_filename(filename: str, shape: typing.Tuple[int, int]) -> str:
    basefolder, basename = os.path.split(filename)
    return f"{basefolder}/{shape[0]}x{shape[1]}/{basename}"


def main(json_filename: str, shape: typing.Tuple[int, int]) -> None:
    output_space = NestedSpace(json_filename)

    new_json_filename = get_new_filename(json_filename, shape)
    new_dirname = os.path.dirname(new_json_filename)

    json_data = output_space.get_json_data()

    logger.set_iterator(len(json_data), percentage_threshold=1)
    for i, entry in enumerate(json_data):
        old_filename = entry['filename']

        old_basefolder = os.path.basename(os.path.dirname(old_filename))
        old_basename = os.path.basename(old_filename)
        new_filename = f"{new_dirname}/{old_basefolder}/{old_basename}"

        json_data[i]['filename'] = new_filename
        if os.path.exists(new_filename):
            continue
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)

        sample = Sample(old_filename, shape)
        try:
            im = sample.imread(old_filename)
        except OSError as e:
            continue
        else:
            im = (im * 255).astype(np.uint8)
            PIL.Image.fromarray(im).save(new_filename)

        logger.iterate()

    with open(new_json_filename, 'w') as f:
        json.dump(json_data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    # python src/scripts/utils/resize_nested_dataset.py  .tmp/dataset/hierarchical/data.json 256 256
    main(sys.argv[1], (int(sys.argv[2]), int(sys.argv[3])))
