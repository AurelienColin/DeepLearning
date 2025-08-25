import glob
import typing

import numpy as np
from rignak.src.logging_utils import logger

from src.generators.base_generators import BatchGenerator
from src.samples.image_to_image.foreground_sample import ForegroundSample
from src.output_spaces.output_space import OutputSpace


class ForegroundGenerator(BatchGenerator):
    def __init__(
            self,
            patterns: typing.Sequence[str],
            batch_size: int,
            shape: typing.Tuple[int, int, int]
    ):
        foreground_filenames = [filename for pattern in patterns for filename in
                                glob.glob(f"{pattern}/foreground/*.??g")]
        logger(f"{len(foreground_filenames)=}")

        self.foreground_filenames: typing.Sequence[str] = np.array(foreground_filenames)
        self.batch_size: int = batch_size
        self.shape: typing.Tuple[int, int, int] = shape
        self.output_space: typing.Optional[OutputSpace] = None

    def __iter__(self):
        return self

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        batch_filenames = np.random.choice(self.foreground_filenames, self.batch_size)
        return self.batch_processing(batch_filenames)

    def reader(self, foreground_filename: str) -> ForegroundSample:
        return ForegroundSample(foreground_filename, self.shape)
