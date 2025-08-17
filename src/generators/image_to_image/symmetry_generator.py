import typing

import numpy as np

from src.generators.base_generators import PostProcessGenerator


class VerticalSymmetryGenerator(PostProcessGenerator):
    def __call__(self, inputs: np.ndarray, outputs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        directions = np.random.randint(0, 1, inputs.shape[0]) * 2 - 1
        for i, direction in enumerate(directions):
            inputs[i] = inputs[i, :, ::direction]
            if outputs.ndim == inputs.ndim:
                outputs[i] = outputs[i, :, ::direction]
        return inputs, outputs
