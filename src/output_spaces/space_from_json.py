import json
import os

from src.output_spaces.output_space import OutputSpace


class TaggerSpace(OutputSpace):

    def setup(self) -> None:
        if not self.sources:
            # Initialize to an empty state
            self._tag_names = []
            self._tags = {}
            self._filename_to_tags = {}
            self._filenames = []
            # _n and _class_weights are LazyProperties and will reflect this empty state.
            return

        json_filename = self.sources[0]
        dirname = os.path.dirname(json_filename)
        with open(json_filename, 'r') as file:
            json_data = json.load(file)

        dataset_size = len(json_data)
        data = [
            (os.path.join(dirname, filename), tags)
            for filename, tags in json_data.items()
        ]
        self.common_setup(dataset_size, data)
