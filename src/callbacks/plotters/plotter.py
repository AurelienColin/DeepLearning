import typing
from dataclasses import dataclass

import numpy as np
from rignak.custom_display import Display
from rignak.lazy_property import LazyProperty

from src.models.model_wrapper import ModelWrapper


def reset_display(method):
    def wrapper(instance, *args, **kwargs):
        instance._display = None
        return method(instance, *args, **kwargs)

    return wrapper


@dataclass
class Plotter:
    model_wrapper: ModelWrapper
    ncols: int
    nrows: int
    thumbnail_size: typing.Tuple[int, int] = (4, 4)

    _display: typing.Optional[Display] = None

    @LazyProperty
    def display(self) -> Display:
        return Display(ncols=self.ncols, nrows=self.nrows, figsize=self.thumbnail_size)

    @staticmethod
    def concatenate(*arrays: np.ndarray, axis: int = 2) -> np.ndarray:
        new_arrays = []
        for array in arrays:
            if array.ndim == 3:
                array = array[:, :, :, np.newaxis]
            if array.shape[3] == 1:
                array = np.tile(array, (1, 1, 1, 3))
            if array.shape[3] > 3:
                array = array[:, :, :, :3]
            new_arrays.append(array)
        new_array = np.concatenate(new_arrays, axis=axis)
        new_array = np.clip(new_array, 0, 1)
        return new_array

    def imshow(self, index: int, image: np.ndarray, **kwargs) -> None:
        kwargs = dict(vmin=0, vmax=255, grid=False, axis_display=False, colorbar_display=False, **kwargs)
        self.display[index].imshow(image[:, :, :3], **kwargs)

    def __call__(self):
        raise NotImplementedError
