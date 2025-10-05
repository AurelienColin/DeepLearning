import os

from src.output_spaces.output_space import OutputSpace
import typing
from rignak.src.lazy_property import LazyProperty


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
            levels = []
            dirname = os.path.dirname(filename)
            for i in range(self.level):
                dirname, basename = os.path.split(dirname)
                levels.append(basename.split(', '))
            data[filename] = levels
        self._data = data
        return self.data
