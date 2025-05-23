import typing

import numpy as np

from src.generators.base_generators import PostProcessGenerator


class VerticalSymmetryGenerator(PostProcessGenerator):

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        inputs, outputs = next(self.generator)
        directions = np.random.randint(0, 1, inputs.shape[0]) * 2 - 1
        for i, direction in enumerate(directions):
            inputs[i] = inputs[i, :, ::direction]
            if outputs.ndim == inputs.ndim:
                outputs[i] = outputs[i, :, ::direction]
        return inputs, outputs
