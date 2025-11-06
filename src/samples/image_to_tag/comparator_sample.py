import numpy as np
from rignak.src.lazy_property import LazyProperty

from src.samples.sample import Sample
import typing
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger
from src.output_spaces.comparator_from_filesystem import ComparatorSpace

class ComparatorSample(Sample):
    def __init__(self, output_space: ComparatorSpace, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_space: ComparatorSpace = output_space

    @LazyProperty
    def input_data(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        return tuple([self.get_input_data(filename) for filename in self.input_filename])


    @LazyProperty
    def output_data(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = np.zeros(self.output_space.level)
        for level in range(self.output_space.level):
            tags1 = self.output_space.data[self.input_filename[0]][level]
            tags2 = self.output_space.data[self.input_filename[1]][level]

            if set(tags1).intersection(tags2):
                outputs[level] = 1

        return outputs, *self.input_data
