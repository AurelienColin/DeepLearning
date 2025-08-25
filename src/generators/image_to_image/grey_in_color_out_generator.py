import typing

import numpy as np

from src.generators.base_generators import PostProcessGenerator


class GreyInColorOutGenerator(PostProcessGenerator):

    def __call__(self, inputs: np.ndarray, outputs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

        inputs = np.mean(inputs, axis=3, keepdims=True)
        return inputs, outputs
