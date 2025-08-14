from typing import Any

from ML.src.output_spaces.custom import hierarchical_tags
from ML.src.output_spaces.custom.sample import Sample
import typing
from dataclasses import dataclass
import numpy as np
from numpy import ndarray, dtype
from rignak.lazy_property import LazyProperty
from ML.src.output_spaces.custom import hierarchical_tags


@dataclass
class HierarchicalSpace:
    json_filename: typing.Optional[str]

    _output: typing.Optional[typing.Tuple[np.ndarray]] = None
    _samples: typing.Optional[typing.List[Sample]] = None

    @property
    def n_outputs(self) -> int:
        return len(hierarchical_tags.categories)

    def setup(self) -> None:
        self._samples = ...

    def setup_from_samples(self, samples: typing.List[Sample]) -> None:
        self._samples = samples



    @LazyProperty
    def output(self) -> typing.Tuple[ndarray]:
        output = [[] for _ in range(self.n_outputs)]
        for sample in self._samples:
            for i, array in enumerate(sample.output):
                output[i].append(array)


        return tuple(np.array(category_output) for category_output in output)
