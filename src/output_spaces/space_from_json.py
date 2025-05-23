import json
import os

from src.output_spaces.output_space import OutputSpace


class TaggerSpace(OutputSpace):

    def setup(self) -> None:
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
