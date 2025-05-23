from src.generators.image_to_tag.classification_generator import ClassificationGenerator
from src.output_spaces.custom.rating_as_float_space import RatingAsFloatSpace


class RatingAsFloatGenerator(ClassificationGenerator):

    def set_output_space(self):
        self.output_space = RatingAsFloatSpace(self.filenames, enforced_tag_names=self.enforced_tag_names)
