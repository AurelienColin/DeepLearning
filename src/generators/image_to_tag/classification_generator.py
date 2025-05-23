import typing

from src.generators.base_generators import BatchGenerator
from src.output_spaces.output_space import OutputSpace
from src.output_spaces.space_from_filesystem import CategorizationSpace
from src.output_spaces.space_from_json import TaggerSpace
from src.samples.image_to_tag.image_to_tag_sample import ImageToTagSample


class ClassificationGenerator(BatchGenerator):
    def __init__(self, *args, enforced_tag_names: typing.Optional[typing.Sequence[str]] = None, **kwargs):
        self.output_space: typing.Optional[OutputSpace] = None
        self.enforced_tag_names: typing.Optional[typing.Sequence[str]] = enforced_tag_names
        super().__init__(*args, **kwargs)

        self.set_output_space()
        self.filenames = self.output_space.filenames

    def set_output_space(self):
        if len(self.filenames) > 1:
            output_space = CategorizationSpace(self.filenames, enforced_tag_names=self.enforced_tag_names)
        else:
            output_space = TaggerSpace(self.filenames, enforced_tag_names=self.enforced_tag_names)
        self.output_space = output_space

    def reader(self, input_filename: str) -> ImageToTagSample:
        return ImageToTagSample(
            output_space=self.output_space,
            input_filename=input_filename,
            shape=self.shape
        )
