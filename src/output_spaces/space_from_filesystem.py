import os

from src.output_spaces.output_space import OutputSpace


class CategorizationSpace(OutputSpace):

    def setup(self) -> None:
        dataset_size = len(self.sources)
        data = [
            (filename, [os.path.basename(os.path.dirname(filename))])
            for filename in self.sources
        ]
        self.common_setup(dataset_size, data)
