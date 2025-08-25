import typing
from dataclasses import dataclass


@dataclass
class Tag:
    index: int
    name: str
    dataset_size: int
    number_of_use: int = 0
    _frequency: typing.Optional[float] = None

    @property
    def frequency(self) -> float:
        return self._frequency

    def set_frequency(self, denominator: int) -> None:
        self._frequency = self.number_of_use / self.dataset_size
