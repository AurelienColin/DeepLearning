import os

from src.output_spaces.output_space import OutputSpace
import typing
from rignak.src.lazy_property import LazyProperty
import functools
import glob
import numpy as np


@functools.cache
def cached_glob(pattern):
    return glob.glob(pattern)


class ComparatorSpace(OutputSpace):
    level: int = 2

    _data: typing.Optional[typing.Dict[str, typing.Sequence[typing.Sequence[str]]]] = None

    @property
    def filenames(self) -> typing.List[str]:
        return list(self.data.keys())

    @LazyProperty
    def data(self) -> typing.Dict[str, typing.Sequence[typing.Sequence[str]]]:
        return self.setup()

    def setup(self) -> typing.Dict[str, typing.Sequence[typing.Sequence[str]]]:
        data = {}
        for filename in self.sources:
            if ', ' in filename:
                continue
            levels = []
            dirname = os.path.dirname(filename)
            for _ in range(self.level):
                dirname, basename = os.path.split(dirname)
                levels.append(basename.split(','))
            data[filename] = levels
        self._data = data
        return self.data

    def get_sisters(self, filename: str, level: int) -> typing.List[str]:
        split_name = filename.split('\\')

        for current_level in range(self.level + 1):
            i = -current_level - 1
            if current_level != level:
                split_name[i] = '*'
            else:
                partial_name = np.random.choice(split_name[i].split(','))
                split_name[i] = f"*{partial_name}*"
        split_name[-1] = '*'

        pattern = "\\".join(split_name)
        candidates = cached_glob(pattern)
        candidates = [
            candidate
            for candidate in candidates
            if candidate in self.data and ', ' not in candidate
        ]
        if not candidates:
            print(f"{filename=}")
            print(f"{pattern=}")
        return candidates
