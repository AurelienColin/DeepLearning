from src.output_spaces.custom.nested.sample import Sample
from src.output_spaces.custom.nested.nested_tags import categories, Category
import typing
from dataclasses import dataclass
import numpy as np
from rignak.src.lazy_property import LazyProperty
import json
from rignak.src.logging_utils import logger

@dataclass
class NestedSpace:
    json_filename: typing.Optional[str]

    _samples: typing.Optional[typing.Dict[str, Sample]] = None
    _size: typing.Optional[int] = None
    _class_weights: typing.Optional[np.ndarray] = None

    def __len__(self) -> int:
        return sum((len(category) for category in self.categories))

    @property
    def categories(self) -> typing.List[Category]:
        return categories

    @property
    def labels(self) -> typing.List[str]:
        return [category.name for category in self.categories]

    @property
    def filenames(self) -> typing.List[str]:
        return list(self.samples.keys())

    @LazyProperty
    def class_weights(self) -> np.ndarray:
        logger("Compute class weight", indent=1)
        weights = np.zeros(len(self))

        logger.set_iterator(len(self.samples), percentage_threshold=5)
        for sample in self.samples.values():
            weights += sample.output
            logger.iterate()

        labels = [label for category in self.categories for label in category.labels]
        for label, weight in zip(labels, weights):
            if not weight:
                logger(f"No sample for `{label}`", level="warning")
        logger("Compute class weight OK", indent=-1)

        weights = np.clip(len(self.samples)/weights, 1E-5, 1E4)
        return weights

    @property
    def n_outputs(self) -> int:
        return len(self.categories)

    @LazyProperty
    def samples(self) -> typing.Dict[str, Sample]:
        samples: typing.Dict[str, Sample] = {}
        json_data = self.get_json_data()

        for entry in json_data:
            sample = Sample(_filename=entry['filename'], _tags=entry['tags'])
            samples[sample._filename] = sample
        return samples

    def get_json_data(self) -> typing.List[typing.Dict[str, str]]:
        with open(self.json_filename) as file:
            json_data = json.load(file)
        return json_data


    def get_array(self, filename: str) -> np.ndarray:
        if filename in self.samples:
            return self.samples[filename].output
        return False
