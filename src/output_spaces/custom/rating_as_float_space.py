import numpy as np

from src.output_spaces.space_from_json import TaggerSpace


class RatingAsFloatSpace(TaggerSpace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n: int = 1
        self.class_weights = None

    def get_array(self, filename: str) -> np.ndarray:
        array = np.empty(1)
        tag = self.filename_to_tags[filename][0]
        array[0] = {
            "rating:g": 0.,
            "rating:s": 0.33,
            "rating:q": 0.66,
            "rating:e": 1.,
        }[tag.name]
        return array
