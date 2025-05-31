import numpy as np

from src.samples.sample import Sample
from rignak.lazy_property import LazyProperty


class ImageToImageSample(Sample):
    @LazyProperty
    def output_data(self) -> np.ndarray:
        return self.imread(self.output_filename).copy()
