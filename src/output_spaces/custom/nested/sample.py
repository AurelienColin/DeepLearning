import typing
from src.scripts.utils.download_from_danbooru import JsonDownloader
from src.output_spaces.custom.nested.nested_tags import categories
import numpy as np
from dataclasses import dataclass
from rignak.lazy_property import LazyProperty


@dataclass
class Sample:
    entry: typing.Optional[typing.Dict[str, typing.Any]] = None

    _tags: typing.Optional[str] = None
    _url: typing.Optional[str] = None
    _output: typing.Optional[np.ndarray] = None
    _filename: typing.Optional[str] = None

    @LazyProperty
    def tags(self) -> str:
        return f"{self.entry['tag_string']} rating:{self.entry['rating']}"

    @LazyProperty
    def url(self) -> str:
        url = None
        media_asset = {}
        if 'media_asset' in self.entry and 'variants' in self.entry['media_asset']:
            for media_asset in self.entry['media_asset']['variants']:
                if media_asset['type'] == 'sample':
                    url = media_asset['url']
            if url is None:
                url = media_asset['url']
        return url

    @LazyProperty
    def output(self) -> np.ndarray:
        output = []
        for category in categories:
            category_output = np.zeros(len(category.subcategories))
            res = category.accept(self.tags.split())
            if res is not None:
                category_output[res[0]] = 1
            output.append(category_output)
        return np.concatenate(output)


def get_samples(tags: str, n_entries: int) -> typing.List[Sample]:
    samples = []
    entries = JsonDownloader(tags, n_entries=n_entries).run()
    for entry in entries:
        sample = Sample(entry=entry)
        samples.append(sample)
    return samples
