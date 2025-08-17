import numpy as np

from src.generators.base_generators import PostProcessGenerator


class Normalizer(PostProcessGenerator):
    ITERATIONS: int = 100

    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.means: np.ndarray = np.zeros(channels)
        self.stds: np.ndarray = np.zeros(channels)

        for i in range(self.ITERATIONS):
            inputs, _ = next(self.generator)
            self.means += np.mean(inputs, axis=(0, 1, 2))
        self.means = self.means / self.ITERATIONS

        for i in range(self.ITERATIONS):
            inputs, _ = next(self.generator)
            self.stds += np.mean(np.abs(inputs - self.means), axis=(0, 1, 2))
        self.stds = self.stds / self.ITERATIONS

    def __call__(self, inputs: np.ndarray, outputs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        inputs = (inputs - self.means) / self.stds
        outputs = (outputs - self.means) / self.stds
        return inputs, outputs
