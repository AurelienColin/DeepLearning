import numpy as np
from rignak.src.lazy_property import LazyProperty

from src.output_spaces.output_space import OutputSpace
from src.samples.sample import Sample


class ImageToTagSample(Sample):
    def __init__(self, output_space: OutputSpace, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_space: OutputSpace = output_space

    @LazyProperty
    def output_data(self) -> np.ndarray:
        return self.output_space.get_array(self.input_filename)
