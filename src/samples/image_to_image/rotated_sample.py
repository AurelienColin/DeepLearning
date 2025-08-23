import random
import typing

import PIL.Image
import numpy as np

from src.samples.sample import Sample


class RotatedSample(Sample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maximum_rotation: int = 20
        self.fillcolor: typing.Tuple[int, int, int, int] = (255, 255, 255, 0)

    def get_4d_fill_color(self, array: np.ndarray) -> typing.Sequence[int]:
        return (*self.fillcolor, 0)[:array.shape[2]]

    def rotate_with_pil(
            self,
            im: PIL.Image.Image,
            maximum_angle: typing.Optional[int] = None,
            fillcolor: typing.Optional[int] = None,
    ) -> PIL.Image.Image:
        if maximum_angle is None:
            maximum_angle = self.maximum_rotation
        if fillcolor is None:
            fillcolor = self.fillcolor

        rotation_angle = random.uniform(-maximum_angle, maximum_angle)
        im = im.rotate(rotation_angle, expand=True, fillcolor=fillcolor, resample=PIL.Image.Resampling.BILINEAR)
        return im

    def pad(self, array: np.ndarray[np.float32]) -> np.ndarray:
        x0 = int((self.shape[0] - array.shape[0]) / 2)
        x1 = self.shape[0] - x0 - array.shape[0]

        y0 = int((self.shape[1] - array.shape[1]) / 2)
        y1 = self.shape[1] - y0 - array.shape[1]

        array = np.pad(array, ((x0, x1), (y0, y1), (0, 0)), 'constant', constant_values=-1)
        mask = np.max(array < 0, axis=2)
        array[mask] = self.fillcolor
        return array
