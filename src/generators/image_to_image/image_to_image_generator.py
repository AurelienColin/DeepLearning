import os
from rignak.path import listdir
from src.generators.base_generators import BatchGenerator
from src.samples.image_to_image.image_to_image_sample import ImageToImageSample


class ImageToImageGenerator(BatchGenerator):
    def reader(self, input_filename: str) -> ImageToImageSample:
        folder = os.path.dirname(input_filename)
        input_filename, output_filename = listdir(folder)
        input_filename = os.path.join(folder, os.path.basename(input_filename))
        output_filename = os.path.join(folder, os.path.basename(output_filename))
        return ImageToImageSample(input_filename, self.shape, output_filename=output_filename)
