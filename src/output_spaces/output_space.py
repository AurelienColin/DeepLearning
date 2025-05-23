import os
import typing
from dataclasses import dataclass

import numpy as np
from Rignak.lazy_property import LazyProperty
from Rignak.logging_utils import logger

from src.output_spaces.tag import Tag


@dataclass
class OutputSpace:
    sources: typing.Sequence[str]
    enforced_tag_names: typing.Optional[typing.Sequence[str]] = None
    _n: typing.Optional[int] = None
    limit: int = 100

    _tag_names: typing.Optional[typing.List[str]] = None
    _tags: typing.Optional[typing.Dict[str, Tag]] = None
    _filename_to_tags: typing.Optional[typing.Dict[str, typing.List[Tag]]] = None
    _filenames: typing.Optional[typing.List[str]] = None
    _class_weights: typing.Optional[np.ndarray] = None

    @LazyProperty
    def class_weights(self) -> np.ndarray:
        weights = np.array([self.n / tag.number_of_use for tag in self.tags.values()])
        weights = weights / np.nanmean(weights)
        return weights

    @LazyProperty
    def n(self) -> int:
        return len(self.tags)

    def get_array(self, filename: str) -> np.ndarray:
        hot_encoded = np.zeros(self.n)
        for tag in self.filename_to_tags[filename]:
            hot_encoded[tag.index] = 1
        return hot_encoded

    @LazyProperty
    def filenames(self) -> typing.Sequence[str]:
        self.setup()
        return self._filenames

    @LazyProperty
    def tags(self) -> typing.Dict[str, Tag]:
        self.setup()
        return self._tags

    @LazyProperty
    def tag_names(self) -> typing.List[str]:
        self.setup()
        return self._tag_names

    @LazyProperty
    def filename_to_tags(self) -> typing.Dict[str, typing.List[Tag]]:
        self.setup()
        return self._filename_to_tags

    def setup(self) -> None:
        raise NotImplementedError

    def common_setup(
            self,
            dataset_size: int,
            data: typing.Iterable[typing.Tuple[str, typing.List[str]]]
    ) -> None:
        logger(f"Setup output space for {dataset_size} files.", indent=1)
        tags: typing.Dict[str, Tag] = {}
        tag_names: typing.List[str] = []
        filename_to_tags: typing.Dict[str, typing.List[Tag]] = {}

        for filename, tag_names_list in data:
            if not os.path.exists(filename) or filename[-3:] not in ('jpg', 'png'):
                continue
            filename_to_tags[filename] = []
            for tag_name in tag_names_list:
                if self.enforced_tag_names is not None and tag_name not in self.enforced_tag_names:
                    continue
                if tag_name not in tags:
                    tag = Tag(index=len(tags), name=tag_name, dataset_size=dataset_size)
                    tag_names.append(tag.name)
                    tags[tag_name] = tag
                else:
                    tag = tags[tag_name]
                tag.number_of_use += 1
                filename_to_tags[filename].append(tag)

        self._tag_names = tag_names
        self._filename_to_tags = filename_to_tags
        self._tags = tags
        self._filenames = list(self._filename_to_tags.keys())
        self.sort_tags()
        logger(f"Found {self.n} tags.")
        logger(f"Setup output space OK", indent=-1)

    def sort_tags(self) -> None:
        tag_names = [
            name for number_of_use, name in sorted(
                [(tag.number_of_use, tag.name) for tag in self._tags.values()],
                reverse=True)[:self.limit]
        ]
        self._tag_names = tag_names
        for i, tag_name in enumerate(tag_names):
            self._tags[tag_name].index = i
