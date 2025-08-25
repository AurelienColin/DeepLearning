import typing

import numpy as np
from scipy import ndimage

from src.generators.base_generators import PostProcessGenerator


class BlurryGenerator(PostProcessGenerator):
    SIGMAX_MAX = 1

    def blur(self, array: np.ndarray, sigma: float) -> np.ndarray:
        array = np.clip(
            ndimage.gaussian_filter(array, sigma,axes=(0, 1)),
            np.min(array),
            np.max(array)
        )
        return array

    def __call__(self, inputs: np.ndarray, outputs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        sigmas = np.random.exponential(self.SIGMAX_MAX, size=inputs.shape[0])
        inputs = np.stack([self.blur(array, sigma) for array, sigma in zip(inputs, sigmas)], axis=0)
        outputs = np.stack([self.blur(array, sigma) for array, sigma in zip(outputs, sigmas)], axis=0)

        sigmas = np.clip(sigmas/4, 0, 1)
        sigmas = sigmas[:, np.newaxis, np.newaxis, np.newaxis]
        sigmas = np.tile(sigmas, (1, *inputs.shape[1:3], 1))
        outputs = np.concatenate((outputs, sigmas), axis=3)
        return inputs, outputs
