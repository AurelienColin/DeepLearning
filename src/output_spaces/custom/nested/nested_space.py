from typing import Any

from src.output_spaces.custom.nested.sample import Sample
from src.output_spaces.custom.nested.nested_tags import categories, Category
import typing
from dataclasses import dataclass
import numpy as np
from rignak.lazy_property import LazyProperty
import json


@dataclass
class NestedSpace:
    json_filename: typing.Optional[str]

    _samples: typing.Optional[typing.Dict[str, Sample]] = None
    _size: typing.Optional[int] = None

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

    @property
    def class_weights(self) -> np.ndarray:
        return np.ones(len(self))

    @property
    def n_outputs(self) -> int:
        return len(self.categories)

    @LazyProperty
    def samples(self) -> typing.Dict[str, Sample]:
        samples: typing.Dict[str, Sample] = {}

        with open(self.json_filename) as file:
            json_data = json.load(file)

        for entry in json_data:
            sample = Sample(_filename=entry['filename'], _tags=entry['tags'])
            samples[sample._filename] = sample
        return samples

    def get_array(self, filename: str) -> np.ndarray:
        return self.samples[filename].output
