import os
import tempfile
import unittest
from typing import Any, Literal, Tuple

import numpy as np
from PIL import Image
import cv2 # Added for interpolation flags

# Import the actual Sample class
from src.samples.sample import Sample

# Helper function to create dummy images, now at module level
def create_dummy_image(
    tmpdir: str,
    filename: str,
    size: Tuple[int, int],
    mode: Literal['L', 'RGB', 'RGBA'],
    color: Any = 'white'
) -> str:
    filepath = os.path.join(tmpdir, filename)
    img = None
    if color == "gradient":
        width, height = size
        array = np.zeros((height, width, len(mode) if mode != 'L' else 1), dtype=np.uint8)
        for x in range(width):
            val = int((x / width) * 255)
            if mode == 'L':
                array[:, x] = val
            elif mode == 'RGB':
                array[:, x, :] = [val, (255 - val) % 255, (x * 2) % 255] # Simple gradient
            elif mode == 'RGBA':
                 array[:, x, :] = [val, (255 - val) % 255, (x * 2) % 255, 255]
        img = Image.fromarray(array, mode if len(mode) != 1 else None) # PIL mode for L is None if single channel from array
    else:
        if mode == 'L':
            img = Image.new('L', size, color=color)
        elif mode == 'RGB':
            img = Image.new('RGB', size, color=color)
        elif mode == 'RGBA':
            img = Image.new('RGBA', size, color=color)
    
    if img:
        img.save(filepath, 'PNG')
    else:
        raise ValueError(f"Could not create image for mode {mode} and color {color}")
    return filepath

class TestSample(unittest.TestCase):
    def test_imread_rgb_basic_resize_and_normalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_rgb_path = create_dummy_image(tmpdir, "dummy_rgb.png", (64, 64), 'RGB', color='red')
            sample_rgb = Sample(input_filename=dummy_rgb_path, shape=(32, 32, 3))
            input_data = sample_rgb.input_data
            self.assertIsInstance(input_data, np.ndarray)
            self.assertEqual(input_data.shape, (32, 32, 3))
            self.assertEqual(input_data.dtype, np.float32)
            self.assertTrue(np.all((input_data >= 0) & (input_data <= 1)))

    def test_imread_rgba_to_rgb_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_rgba_path = create_dummy_image(tmpdir, "dummy_rgba.png", (48, 48), 'RGBA', color=(0, 255, 0, 128))
            sample_rgba_to_rgb = Sample(input_filename=dummy_rgba_path, shape=(48, 48, 3))
            input_data = sample_rgba_to_rgb.input_data
            self.assertEqual(input_data.shape, (48, 48, 3))
            self.assertEqual(input_data.dtype, np.float32)
            self.assertTrue(np.all((input_data >= 0) & (input_data <= 1)))

    def test_imread_grayscale_to_3_channel_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_l_path_to_rgb = create_dummy_image(tmpdir, "dummy_l_to_rgb.png", (32, 32), 'L', color='grey')
            sample_l_to_rgb = Sample(input_filename=dummy_l_path_to_rgb, shape=(32, 32, 3))
            input_data = sample_l_to_rgb.input_data
            self.assertEqual(input_data.shape, (32, 32, 3))
            self.assertEqual(input_data.dtype, np.float32)
            self.assertTrue(np.all((input_data >= 0) & (input_data <= 1)))

    def test_imread_grayscale_to_1_channel_no_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_l_path_to_l = create_dummy_image(tmpdir, "dummy_l_to_l.png", (32, 32), 'L', color='black')
            sample_l_to_l = Sample(input_filename=dummy_l_path_to_l, shape=(32, 32, 1))
            input_data = sample_l_to_l.input_data
            self.assertEqual(input_data.shape, (32, 32, 1))
            self.assertEqual(input_data.dtype, np.float32)
            self.assertTrue(np.all((input_data >= 0) & (input_data <= 1)))

    def test_imread_no_resize_if_dimensions_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_rgb_no_resize_path = create_dummy_image(tmpdir, "dummy_rgb_no_resize.png", (40, 40), 'RGB', color='blue')
            sample_no_resize = Sample(input_filename=dummy_rgb_no_resize_path, shape=(40, 40, 3))
            input_data = sample_no_resize.input_data
            self.assertEqual(input_data.shape, (40, 40, 3))
            self.assertEqual(input_data.dtype, np.float32)
            self.assertTrue(np.all((input_data >= 0) & (input_data <= 1)))

    def test_imread_interpolation_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            original_size = (10, 10)
            target_size = (20, 20)
            target_shape = (target_size[1], target_size[0], 3) # H, W, C
            # Use "gradient" color to ensure interpolation differences
            dummy_image_path = create_dummy_image(tmpdir, "interp_test.png", original_size, 'RGB', color='gradient')

            interpolations = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
            }
            
            results: Dict[str, np.ndarray] = {}

            for name, flag in interpolations.items():
                sample = Sample(
                    input_filename=dummy_image_path,
                    shape=target_shape,
                    interpolation=flag
                )
                input_data = sample.input_data
                self.assertEqual(input_data.shape, target_shape, f"Shape mismatch for {name} interpolation")
                self.assertEqual(input_data.dtype, np.float32, f"Dtype mismatch for {name} interpolation")
                results[name] = input_data
            
            # Assert that different interpolations produce different results
            # (excluding cases where image content might lead to identical results by chance,
            # though with a simple green image and resizing, they should differ)
            self.assertFalse(np.array_equal(results["nearest"], results["linear"]), "Nearest and Linear should differ")
            self.assertFalse(np.array_equal(results["linear"], results["cubic"]), "Linear and Cubic should differ")
            self.assertFalse(np.array_equal(results["nearest"], results["cubic"]), "Nearest and Cubic should differ")

    def test_input_data_caching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_image_path = create_dummy_image(tmpdir, "caching_test.png", (16, 16), 'RGB', color='blue')
            sample = Sample(input_filename=dummy_image_path, shape=(16, 16, 3))

            # Access data for the first time
            data1 = sample.input_data
            self.assertTrue("_lazy_input_data" in sample.__dict__, "LazyProperty cache attribute not found.")

            # Modify the underlying file
            create_dummy_image(tmpdir, "caching_test.png", (16, 16), 'RGB', color='red') # Overwrite with different color

            # Access data again
            data2 = sample.input_data
            
            # data1 and data2 should be identical if caching works
            np.testing.assert_array_equal(data1, data2, "Cached data did not match initial data after file modification.")

    def test_imread_non_existent_file(self) -> None:
        non_existent_path = "path/to/a/completely/non_existent_image.png"
        sample = Sample(input_filename=non_existent_path, shape=(32, 32, 3))
        
        with self.assertRaises(FileNotFoundError):
            _ = sample.input_data


if __name__ == '__main__':
    unittest.main()
