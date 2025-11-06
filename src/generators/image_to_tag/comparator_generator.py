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
            enforced_tag_names: typing.Optional[typing.Sequence[str]] = None,
            output_space: typing.Optional[ComparatorSpace] = None,
            **kwargs
    ):
        self.enforced_tag_names: typing.Optional[typing.Sequence[str]] = enforced_tag_names
        super().__init__(*args, **kwargs)
        self.output_space: typing.Optional[ComparatorSpace] = output_space
        self.filenames: typing.Sequence[str] = list(self.output_space.data.keys())

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        first_filenames = np.random.choice(self.filenames, self.batch_size, replace=False)
        levels = np.random.randint(0, self.output_space.level+1, self.batch_size)
        second_filenames = [
            np.random.choice(self.output_space.get_sisters(filename, level))
            for filename, level in zip(first_filenames, levels)
        ]

        batch_filenames = zip(first_filenames, second_filenames)
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
