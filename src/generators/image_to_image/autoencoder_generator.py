from src.generators.base_generators import BatchGenerator
from src.samples.image_to_image.autoencoder_sample import AutoEncodingSample


class AutoEncoderGenerator(BatchGenerator):
    def reader(self, input_filename: str) -> AutoEncodingSample:
        return AutoEncodingSample(input_filename, self.shape)
