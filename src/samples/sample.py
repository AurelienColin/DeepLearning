import typing
from dataclasses import dataclass

import PIL.Image
import cv2
import numpy as np
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger

@dataclass
class Sample:
    input_filename: str
    shape: typing.Sequence[int]

    output_filename: typing.Optional[str] = None
    interpolation: int = cv2.INTER_LINEAR

    _input_data: typing.Optional[np.ndarray] = None
    _output_data: typing.Optional[np.ndarray] = None

    def get_input_data(self, filename: str) -> np.ndarray:
        try:
            im = self.imread(filename)
        except OSError as e:
            logger(
                f"Error reading `{self.input_filename}`: `{e}`. "
                f"Continue with zero-filled array of shape {self.shape}",
                level='error'
            )
            im = np.zeros(self.shape)
        return im

    @LazyProperty
    def input_data(self) -> np.ndarray:
        return self.get_input_data(self.input_filename)

    @LazyProperty
    def output_data(self) -> np.ndarray:
        raise NotImplementedError

    def imread(self, path: str, resize: bool = True) -> np.ndarray:
        array = np.array(PIL.Image.open(path)).astype(np.float32) / 255
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        if array.ndim == 2 and len(self.shape) == 3:
            array = np.expand_dims(array, axis=-1)
            array = np.repeat(array, self.shape[2], axis=2)

        if resize and array.shape[:2] != self.shape[:2]:
            array = cv2.resize(array, self.shape[:2][::-1], interpolation=self.interpolation)

        return array


