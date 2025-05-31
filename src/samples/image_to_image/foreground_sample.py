import numpy as np
import scipy.ndimage
from PIL import Image
from rignak.lazy_property import LazyProperty

from src.samples.image_to_image.rotated_sample import RotatedSample


class ForegroundSample(RotatedSample):
    def __init__(self, filename: str, *args, **kwargs):
        super().__init__(filename, *args, **kwargs)
        self.filename: str = filename

    @LazyProperty
    def input_data(self) -> np.ndarray:
        self.setup()
        return self._input_data

    @LazyProperty
    def output_data(self) -> np.ndarray:
        self.setup()
        return self._output_data

    def setup(self) -> None:
        print(f"{self.filename=}")
        foreground = Image.open(self.filename)
        assert foreground.mode == "RGBA"

        foreground = self.rotate_with_pil(foreground)

        scale_factor = min(self.shape[0] / foreground.size[1], self.shape[1] / foreground.size[0])
        print(self.shape[0] / foreground.size[0])
        print(self.shape[1] / foreground.size[1])

        new_size = (int(foreground.size[0] * scale_factor), int(foreground.size[1] * scale_factor))
        print(new_size)
        foreground = np.array(foreground.resize(new_size, Image.ANTIALIAS)).astype(np.float32) / 255
        foreground = self.pad(foreground)

        _input_data = foreground[:, :, :3]

        _output_data = np.where(foreground[:, :, 3] > .5, 1., 0.)

        _input_data[_output_data < 0.5] = self.fillcolor[:3]
        _output_data = scipy.ndimage.binary_opening(_output_data, iterations=1, structure=np.ones((3, 3)))

        self._input_data = _input_data
        self._output_data = _output_data


if __name__ == "__main__":
    # python src/samples/image_to_image/foreground_sample.py
    import matplotlib.pyplot as plt

    sample = ForegroundSample(
        "~/Documents/E/datasets/overlay/validation/foreground/1798118_processed.png",
        (384, 256)
    )

    plt.figure()
    plt.subplot(121)
    plt.imshow(sample.input_data)
    plt.subplot(122)
    plt.imshow(sample.output_data)
    plt.show()
