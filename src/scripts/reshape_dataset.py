import argparse
import os
import shutil
import typing
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reshape each image in a dataset.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing the images to reshape.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to contain the reshaped images.")
    parser.add_argument("--width", type=int, default=96, help="Width of the reshaped images.")
    parser.add_argument("--height", type=int, default=96, help="Height of the reshaped images.")
    return parser.parse_args()


@dataclass
class Processor:
    input_folder: str
    output_folder: str
    width: int
    height: int

    extensions: typing.Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
    _input_filenames: typing.Optional[typing.Sequence[Path]] = None

    def run_on_image(self, input_filename: Path, output_filename: Path) -> None:

        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(input_filename) as img:
            img.thumbnail((self.width, self.height), Image.ANTIALIAS)
            delta_w = self.width - img.size[0]
            delta_h = self.height - img.size[1]
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            img_padded = ImageOps.expand(img, padding, fill=(0, 0, 0))
            img_padded.save(output_filename)

    @LazyProperty
    def input_filenames(self) -> typing.Sequence[Path]:
        filenames = []
        for root, _, basenames in os.walk(self.input_folder):
            root = Path(root)
            for basename in basenames:
                filename = root / basename
                if filename.suffix.lower() in self.extensions:
                    filenames.append(filename)
        return filenames

    def run(self) -> None:
        logger.set_iterator(len(self.input_filenames), percentage_threshold=2)

        to_remove = []
        for input_filename in self.input_filenames:
            logger.iterate(str(input_filename))

            relative_path = input_filename.relative_to(self.input_folder)
            output_filename = Path(self.output_folder) / relative_path

            try:
                self.run_on_image(input_filename, output_filename)
            except TypeError:
                to_remove.append(output_filename.parent)

        for folder in set(to_remove):
            shutil.rmtree(folder)

    @classmethod
    def static_run(cls) -> None:
        args = get_args()
        processor = cls(args.input_folder, args.output_folder, args.width, args.height)
        processor.run()


if __name__ == "__main__":
    # python src/scripts/reshape_dataset.py --input_folder "~/Documents/E/datasets/style_transfer" --output_folder "~/Documents/E/datasets/style_transfer128" --width 128 --height 128
    Processor.static_run()
