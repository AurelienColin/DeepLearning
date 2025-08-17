import typing
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import numpy as np

from src.output_spaces.output_space import OutputSpace
from src.samples.sample import Sample


class BatchGenerator:
    def __init__(self, filenames: typing.Sequence[str], batch_size: int, shape: typing.Tuple[int, int, int]):
        self.filenames: typing.Sequence[str] = np.array(filenames)
        self.batch_size: int = batch_size
        self.shape: typing.Tuple[int, int, int] = shape
        self.output_space: typing.Optional[OutputSpace] = None

    def __iter__(self):
        return self

    def reader(self, input_filename: str) -> Sample:
        raise NotImplementedError

    def batch_processing(
            self,
            filenames: typing.Iterable[typing.Tuple[str, str]]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        with ThreadPool(processes=self.batch_size) as pool:
            data = pool.map(self.reader, filenames)

        inputs = np.stack([e.input_data for e in data], axis=0).astype(np.float32)
        outputs = np.stack([e.output_data for e in data], axis=0).astype(np.float32)
        return inputs, outputs

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        batch_filenames = np.random.choice(self.filenames, self.batch_size, replace=False)
        return self.batch_processing(batch_filenames)


@dataclass
class PostProcessGenerator:
    generator: typing.Optional[typing.Union[BatchGenerator, "PostProcessGenerator"]]

    def __iter__(self):
        return self

    def __call__(self, inputs: np.ndarray, outputs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self(*next(self.generator))

    def batch_processing(
            self,
            filenames: typing.Iterable[typing.Tuple[str, str]]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self(*self.generator.batch_processing(filenames))

    @property
    def output_space(self) -> OutputSpace:
        return self.generator.output_space

    @property
    def batch_size(self) -> int:
        return self.generator.batch_size


def compose_generators(
        generator: BatchGenerator,
        composition: typing.Sequence[typing.Type] = ()
) -> BatchGenerator:
    for next_generator in composition:
        generator = next_generator(generator)
    return generator
