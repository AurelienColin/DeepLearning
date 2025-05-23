import typing

import numpy as np

from src.generators.base_generators import PostProcessGenerator


class GreyInColorOutGenerator(PostProcessGenerator):

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        inputs, outputs = next(self.generator)

        inputs = np.mean(inputs, axis=3, keepdims=True)
        return inputs, outputs
