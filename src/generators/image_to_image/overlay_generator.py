import glob
import typing

import numpy as np
from rignak.src.logging_utils import logger

from src.generators.base_generators import BatchGenerator
from src.samples.image_to_image.overlaid_sample import OverlaidSample


class OverlayGenerator(BatchGenerator):

    def __init__(
            self,
            patterns: typing.Sequence[str],
            batch_size: int,
            shape: typing.Tuple[int, int, int]
    ):
        foreground_filenames = [filename for pattern in patterns for filename in
                                glob.glob(f"{pattern}/foreground/*.??g")]
        background_filenames = [filename for pattern in patterns for filename in
                                glob.glob(f"{pattern}/background/*.??g")]
        logger(f"{len(foreground_filenames)=}")
        logger(f"{len(background_filenames)=}")

        self.foreground_filenames: typing.Sequence[str] = np.array(foreground_filenames)
        self.background_filenames: typing.Sequence[str] = np.array(background_filenames)
        self.batch_size: int = batch_size
        self.shape: typing.Tuple[int, int, int] = shape
        self.output_space: typing.Optional[OutputSpace] = None

    def __iter__(self):
        return self

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        batch_foreground_filenames = np.random.choice(self.foreground_filenames, self.batch_size)
        batch_background_filenames = np.random.choice(self.background_filenames, self.batch_size)
        batch_filenames = list(zip(batch_foreground_filenames, batch_background_filenames))
        return self.batch_processing(batch_filenames)

    def reader(self, filenames: typing.Tuple[str, str]) -> OverlaidSample:
        foreground_filename, background_filename = filenames
        return OverlaidSample(foreground_filename, background_filename, self.shape)
