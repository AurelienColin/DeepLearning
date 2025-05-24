import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_image.image_to_image_generator import ImageToImageGenerator
from samples.image_to_image.image_to_image_sample import ImageToImageSample


class TestImageToImageGenerator(unittest.TestCase):

    def test_initialization(self):
        # Arrange
        shape = (128, 128, 3)
        batch_size = 4
        samples = ["path/to/sample1_folder", "path/to/sample2_folder"] # These are folders

        # Act
        generator = ImageToImageGenerator(shape=shape, batch_size=batch_size, samples=samples)

        # Assert
        self.assertEqual(generator.shape, shape)
        self.assertEqual(generator.batch_size, batch_size)
        self.assertEqual(generator.samples, samples)
        self.assertEqual(generator.input_shape, shape)
        self.assertEqual(generator.output_shape, shape)


    @patch('os.listdir')
    @patch('os.path.dirname')
    @patch('samples.image_to_image.image_to_image_sample.ImageToImageSample')
    def test_reader(self, MockImageToImageSample, mock_os_path_dirname, mock_os_listdir):
        # Arrange
        mock_sample_instance = MockImageToImageSample.return_value
        generator = ImageToImageGenerator(shape=(128, 128, 3), batch_size=4, samples=[])
        
        test_input_filename_placeholder = "path/to/some_folder/any_file.png" # The actual file name in folder doesn't matter for this test
        folder_path = "path/to/some_folder"
        
        mock_os_path_dirname.return_value = folder_path
        # os.listdir should return two filenames, which the reader will sort
        # to determine input and output.
        mock_os_listdir.return_value = ["output_img.png", "input_img.jpg"] 
        
        expected_input_path = os.path.join(folder_path, "input_img.jpg")
        expected_output_path = os.path.join(folder_path, "output_img.png")

        # Act
        result = generator.reader(test_input_filename_placeholder)

        # Assert
        mock_os_path_dirname.assert_called_once_with(test_input_filename_placeholder)
        mock_os_listdir.assert_called_once_with(folder_path)
        MockImageToImageSample.assert_called_once_with(
            expected_input_path, 
            (128, 128, 3), 
            output_filename=expected_output_path
        )
        self.assertEqual(result, mock_sample_instance)

    @patch('samples.image_to_image.image_to_image_sample.ImageToImageSample')
    @patch('os.listdir') # Mock listdir for reader
    @patch('os.path.dirname') # Mock dirname for reader
    def test_getitem_output_shapes(self, mock_dirname, mock_listdir, MockImageToImageSample):
        # Arrange
        shape = (64, 64, 1)
        batch_size = 2
        # For ImageToImageGenerator, 'samples' are paths to folders, 
        # and reader will use one file from each folder as a key.
        sample_folders = [f"dummy_folder_{i}" for i in range(batch_size)] 
        
        # Mock os.listdir and os.path.dirname for the reader method
        # These will be called by the reader when __getitem__ calls it.
        mock_dirname.side_effect = lambda x: os.path.dirname(x) # Simple passthrough for dirname
        # listdir needs to return two files for each folder.
        # We'll make it specific to the dummy folders for clarity.
        def listdir_side_effect(path):
            if "dummy_folder_" in path:
                return [f"{path}_output.png", f"{path}_input.jpg"]
            return []
        mock_listdir.side_effect = listdir_side_effect

        # Mock the behavior of ImageToImageSample's __getitem__
        mock_sample_instance = MockImageToImageSample.return_value
        mock_sample_instance.__getitem__.return_value = (
            np.random.rand(*shape),  # input_image
            np.random.rand(*shape)   # output_image
        )

        # Initialize generator
        # The 'samples' list for BatchGenerator's __getitem__ will be iterated.
        # Here, these are just keys that reader will use. We provide dummy file paths
        # that will be processed by the mocked reader.
        # The actual content of these paths doesn't matter due to mocking.
        dummy_file_keys_for_reader = [os.path.join(f, "key.any") for f in sample_folders]
        generator = ImageToImageGenerator(shape=shape, batch_size=batch_size, samples=dummy_file_keys_for_reader)
        
        # Replace the reader method with a mock that returns our pre-configured sample instance
        # This is important because the default reader has its own os calls we've already mocked above.
        generator.reader = MagicMock(return_value=mock_sample_instance)


        # Act
        batch_x, batch_y = generator.__getitem__(0) # Get the first batch

        # Assert
        self.assertEqual(batch_x.shape, (batch_size, *shape))
        self.assertEqual(batch_y.shape, (batch_size, *shape))
        
        # Check that reader was called for each sample in the batch
        self.assertEqual(generator.reader.call_count, batch_size)
        for i in range(batch_size):
            # The reader is called with the "sample key" from the samples list
            generator.reader.assert_any_call(dummy_file_keys_for_reader[i])
        
        # Check that __getitem__ on the sample instance was called for each sample processed by reader
        self.assertEqual(mock_sample_instance.__getitem__.call_count, batch_size)


if __name__ == '__main__':
    unittest.main()
