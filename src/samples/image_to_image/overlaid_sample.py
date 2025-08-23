import random

import numpy as np
from PIL import Image
from rignak.lazy_property import LazyProperty
from rignak.logging_utils import logger

from src.samples.image_to_image.rotated_sample import RotatedSample

class OverlaidSample(RotatedSample):
    def __init__(self, foreground_filename: str, background_filename: str, *args, **kwargs):
        super().__init__(foreground_filename, *args, **kwargs)
        self.foreground_filename: str = foreground_filename
        self.background_filename: str = background_filename

    @LazyProperty
    def input_data(self) -> np.ndarray:
        self.setup()
        return self._input_data

    @LazyProperty
    def output_data(self) -> np.ndarray:
        self.setup()
        return self._output_data

    def setup(self) -> None:
        # logger('Convert input arrays to PIL Images')

        foreground = Image.open(self.foreground_filename)
        if foreground.mode not in ('RGB', 'RGBA'):
            foreground = foreground.convert("RGB")

        background = Image.open(self.background_filename)
        if background.mode not in ('RGB', 'RGBA'):
            background = background.convert("RGB")

        # logger('Random rotation up to 20 degrees')
        foreground = self.rotate_with_pil(
            foreground,
            maximum_angle=self.maximum_rotation,
            fillcolor=self.fillcolor
        )

        # logger('Random downscaling')
        minimum_factor = max(self.shape[0] / foreground.size[0], self.shape[1] / foreground.size[1])
        maximum_factor = min(background.size[0] / foreground.size[0], background.size[1] / foreground.size[1])
        if minimum_factor > maximum_factor:
            logger(f"Error from {foreground.size=}, {background.size} "
                   f"({self.foreground_filename}/{self.background_filename})")
            self._input_data = np.zeros((*self.shape[:2], 3))
            self._output_data = np.zeros((*self.shape[:2], 1))
            return

        scale_factor = random.uniform(minimum_factor, maximum_factor)
        new_size = (int(foreground.size[0] * scale_factor), int(foreground.size[1] * scale_factor))
        foreground = foreground.resize(new_size, Image.LANCZOS)

        # logger('Convert foreground to RGBA to handle transparency')
        foreground = foreground.convert("RGBA")
        datas = foreground.getdata()

        # logger('Make white pixels transparent')
        new_data = []
        for item in datas:
            # Change all white (also shades of whites) pixels to transparent
            if item[0] > 250 and item[1] > 250 and item[2] > 250:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        foreground.putdata(new_data)

        # logger('Random position for overlaying')
        x_offset = random.randint(0, background.size[0] - foreground.size[0])
        y_offset = random.randint(0, background.size[1] - foreground.size[1])

        # logger('Overlay foreground on background')
        background.paste(foreground, (x_offset, y_offset), foreground)

        # logger('Crop to the bounding box of the foreground')
        bbox = foreground.getbbox()
        if bbox:
            background = background.crop((x_offset, y_offset, x_offset + bbox[2], y_offset + bbox[3]))
            foreground = foreground.crop((0, 0, 0 + bbox[2], 0 + bbox[3]))

        # logger('Resize to the exact output shape')
        background = background.resize(self.shape[:2], Image.LANCZOS)
        foreground = foreground.resize(self.shape[:2], Image.LANCZOS)

        background = np.array(background)  # [:, :, :3]
        foreground = np.array(foreground)  # [:, :, 3]
        background = background[:, :, :3]
        foreground = foreground[:, :, 3]

        self._input_data = background / 255
        self._output_data = np.where(foreground > 128, 1., 0.)[:, :, np.newaxis]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample = OverlaidSample(
        "~/Documents/E/datasets/overlay_example/foreground/0.jpg",
        "~/Documents/E/datasets/overlay_example/background/0.jpg",
        (256, 512)
    )

    plt.figure()
    plt.subplot(121)
    plt.imshow(sample.input_data)
    plt.subplot(122)
    plt.imshow(sample.output_data)
    plt.show()
