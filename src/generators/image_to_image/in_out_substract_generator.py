import typing

import numpy as np

from src.generators.base_generators import PostProcessGenerator


class InOutSubstractGenerator(PostProcessGenerator):

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        inputs, outputs = next(self.generator)

        monochrome_input = np.mean(inputs, axis=-1)
        # monochrome_input = gaussian_filter(monochrome_input, sigma=1)

        monochrome_output = np.mean(outputs, axis=-1)
        # monochrome_output = gaussian_filter(monochrome_output, sigma=1)

        difference = np.abs(monochrome_output - monochrome_input)[:, :, :, None]
        difference = np.where(difference > 0.2, 1., 0.)
        return inputs, difference
