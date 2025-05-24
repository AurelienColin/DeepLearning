import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_image.foreground_generator import ForegroundGenerator
from samples.image_to_image.foreground_sample import ForegroundSample


class TestForegroundGenerator(unittest.TestCase):

    @patch('glob.glob')
    @patch('Rignak.logging_utils.logger') # Mock the logger
    def test_initialization(self, mock_logger, mock_glob_glob):
        # Arrange
        patterns = ["data/set1", "data/set2"]
        batch_size = 8
        shape = (128, 128, 3)
        
        # Mock glob.glob to return some dummy file paths
        mock_glob_glob.side_effect = [
            ["data/set1/foreground/img1.png", "data/set1/foreground/img2.jpg"],
            ["data/set2/foreground/img3.jpeg"]
        ]
        expected_filenames = [
            "data/set1/foreground/img1.png", 
            "data/set1/foreground/img2.jpg",
            "data/set2/foreground/img3.jpeg"
        ]

        # Act
        generator = ForegroundGenerator(patterns=patterns, batch_size=batch_size, shape=shape)

        # Assert
        mock_glob_glob.assert_has_calls([
            call("data/set1/foreground/*.??g"),
            call("data/set2/foreground/*.??g")
        ])
        np.testing.assert_array_equal(generator.foreground_filenames, expected_filenames)
        self.assertEqual(generator.batch_size, batch_size)
        self.assertEqual(generator.shape, shape)
        mock_logger.assert_called_once_with(f"{len(expected_filenames)=}")
        # Check default shapes based on BatchGenerator's __init__ if not overridden
        self.assertEqual(generator.input_shape, shape)
        self.assertEqual(generator.output_shape, shape)


    @patch('samples.image_to_image.foreground_sample.ForegroundSample')
    def test_reader(self, MockForegroundSample):
        # Arrange
        mock_sample_instance = MockForegroundSample.return_value
        # Need to initialize with some dummy data for shape, even though it's not directly used by reader
        generator = ForegroundGenerator(patterns=[], batch_size=4, shape=(128, 128, 3)) 
        test_filename = "dummy_foreground.png"

        # Act
        result = generator.reader(test_filename)

        # Assert
        MockForegroundSample.assert_called_once_with(test_filename, (128, 128, 3))
        self.assertEqual(result, mock_sample_instance)

    @patch('numpy.random.choice')
    @patch('samples.image_to_image.foreground_sample.ForegroundSample')
    @patch('glob.glob') # Mock glob for initialization
    @patch('Rignak.logging_utils.logger') # Mock logger for initialization
    def test_next_batch_processing_and_shapes(self, mock_logger, mock_glob, MockForegroundSample, mock_np_choice):
        # Arrange
        patterns = ["data/dummy"]
        shape = (64, 64, 3)
        batch_size = 2
        
        dummy_filenames = ["dummy1.png", "dummy2.png", "dummy3.png"]
        mock_glob.return_value = dummy_filenames # Glob returns these files

        generator = ForegroundGenerator(patterns=patterns, batch_size=batch_size, shape=shape)

        # Mock what np.random.choice will return
        chosen_filenames = [dummy_filenames[0], dummy_filenames[1]]
        mock_np_choice.return_value = chosen_filenames

        # Mock the ForegroundSample instance and its __getitem__
        mock_sample_instance = MockForegroundSample.return_value
        mock_sample_instance.__getitem__.return_value = (
            np.random.rand(*shape),  # input_image (e.g., original image)
            np.random.rand(*shape)   # output_image (e.g., foreground mask)
        )
        # Ensure the reader method returns our mocked sample
        generator.reader = MagicMock(return_value=mock_sample_instance)

        # Act
        batch_x, batch_y = next(generator) # or generator.__next__()

        # Assert
        mock_np_choice.assert_called_once_with(generator.foreground_filenames, batch_size)
        
        self.assertEqual(generator.reader.call_count, batch_size)
        for i in range(batch_size):
            generator.reader.assert_any_call(chosen_filenames[i])
        
        # Check that __getitem__ on the sample was called for each chosen file
        # This depends on how batch_processing is implemented (it calls reader then sample.__getitem__)
        self.assertEqual(mock_sample_instance.__getitem__.call_count, batch_size)

        self.assertEqual(batch_x.shape, (batch_size, *shape))
        self.assertEqual(batch_y.shape, (batch_size, *shape))

    def test_iter(self):
        # Arrange
        generator = ForegroundGenerator(patterns=[], batch_size=4, shape=(128,128,3))
        # Act
        iterator_self = iter(generator)
        # Assert
        self.assertIs(iterator_self, generator)


if __name__ == '__main__':
    unittest.main()
