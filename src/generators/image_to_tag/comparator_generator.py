import typing
import numpy as np
from src.generators.base_generators import BatchGenerator
from src.output_spaces.comparator_from_filesystem import ComparatorSpace
from src.samples.image_to_tag.comparator_sample import ComparatorSample
from multiprocessing.pool import ThreadPool

class ComparatorGenerator(BatchGenerator):
    def __init__(
            self,
            *args,
            output_space: typing.Optional[ComparatorSpace] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_space: typing.Optional[ComparatorSpace] = output_space
        self.filenames: typing.Sequence[str] = list(self.output_space.data.keys())

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        batch_filenames = np.random.choice(self.filenames, (self.batch_size, 2), replace=False)
        return self.batch_processing(batch_filenames)

    def batch_processing(
            self,
            filenames: typing.Iterable[typing.Tuple[str, str]]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        with ThreadPool(processes=self.batch_size) as pool:
            data = pool.map(self.reader, filenames)

        inputs = np.stack([e.input_data for e in data], axis=0).astype(np.float32)
        outputs = [np.stack([e.output_data[i] for e in data], axis=0).astype(np.float32) for i in range(3)]
        return inputs, outputs

    def reader(self, input_filename: str) -> ComparatorSample:
        return ComparatorSample(
            output_space=self.output_space,
            input_filename=input_filename,
            shape=self.shape
        )
