import numpy as np

from src.samples.image_to_image.image_to_image_sample import ImageToImageSample
from Rignak.lazy_property import LazyProperty

class AutoEncodingSample(ImageToImageSample):
    @LazyProperty
    def output_data(self) -> np.ndarray:
        return self.input_data
