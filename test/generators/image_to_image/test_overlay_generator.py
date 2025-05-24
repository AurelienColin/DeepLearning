import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_image.overlay_generator import OverlayGenerator
from samples.image_to_image.overlaid_sample import OverlaidSample


class TestOverlayGenerator(unittest.TestCase):

    @patch('glob.glob')
    @patch('Rignak.logging_utils.logger') # Mock the logger
    def test_initialization(self, mock_logger, mock_glob_glob):
        # Arrange
        patterns = ["data/set1", "data/set2"]
        batch_size = 8
        shape = (128, 128, 3)
        
        # Mock glob.glob to return some dummy file paths
        # It will be called for foregrounds then backgrounds for each pattern
        mock_glob_glob.side_effect = [
            ["data/set1/foreground/fg1.png", "data/set1/foreground/fg2.jpg"], # set1 foregrounds
            ["data/set1/background/bg1.png"],                                # set1 backgrounds
            ["data/set2/foreground/fg3.jpeg"],                               # set2 foregrounds
            ["data/set2/background/bg2.jpg", "data/set2/background/bg3.png"] # set2 backgrounds
        ]
        expected_fg_filenames = [
            "data/set1/foreground/fg1.png", 
            "data/set1/foreground/fg2.jpg",
            "data/set2/foreground/fg3.jpeg"
        ]
        expected_bg_filenames = [
            "data/set1/background/bg1.png",
            "data/set2/background/bg2.jpg",
            "data/set2/background/bg3.png"
        ]

        # Act
        generator = OverlayGenerator(patterns=patterns, batch_size=batch_size, shape=shape)

        # Assert
        mock_glob_glob.assert_has_calls([
            call("data/set1/foreground/*.??g"),
            call("data/set1/background/*.??g"),
            call("data/set2/foreground/*.??g"),
            call("data/set2/background/*.??g"),
        ])
        np.testing.assert_array_equal(generator.foreground_filenames, expected_fg_filenames)
        np.testing.assert_array_equal(generator.background_filenames, expected_bg_filenames)
        self.assertEqual(generator.batch_size, batch_size)
        self.assertEqual(generator.shape, shape)
        
        # Check logger calls
        mock_logger.assert_any_call(f"{len(expected_fg_filenames)=}")
        mock_logger.assert_any_call(f"{len(expected_bg_filenames)=}")

        self.assertEqual(generator.input_shape, shape)
        self.assertEqual(generator.output_shape, shape)


    @patch('samples.image_to_image.overlaid_sample.OverlaidSample')
    def test_reader(self, MockOverlaidSample):
        # Arrange
        mock_sample_instance = MockOverlaidSample.return_value
        # Need to initialize with some dummy data for shape
        generator = OverlayGenerator(patterns=[], batch_size=4, shape=(128, 128, 3)) 
        test_fg_filename = "dummy_fg.png"
        test_bg_filename = "dummy_bg.png"
        test_filenames_tuple = (test_fg_filename, test_bg_filename)

        # Act
        result = generator.reader(test_filenames_tuple)

        # Assert
        MockOverlaidSample.assert_called_once_with(test_fg_filename, test_bg_filename, (128, 128, 3))
        self.assertEqual(result, mock_sample_instance)

    @patch('numpy.random.choice')
    @patch('samples.image_to_image.overlaid_sample.OverlaidSample')
    @patch('glob.glob') # Mock glob for initialization
    @patch('Rignak.logging_utils.logger') # Mock logger for initialization
    def test_next_batch_processing_and_shapes(self, mock_logger, mock_glob, MockOverlaidSample, mock_np_choice):
        # Arrange
        patterns = ["data/dummy"]
        shape = (64, 64, 3)
        batch_size = 2
        
        dummy_fg_files = ["fg1.png", "fg2.png", "fg3.png"]
        dummy_bg_files = ["bg1.png", "bg2.png", "bg3.png"]
        
        # glob is called for fg then bg
        mock_glob.side_effect = [dummy_fg_files, dummy_bg_files]

        generator = OverlayGenerator(patterns=patterns, batch_size=batch_size, shape=shape)

        # Mock what np.random.choice will return
        # First call for foregrounds, second for backgrounds
        chosen_fg_filenames = [dummy_fg_files[0], dummy_fg_files[1]]
        chosen_bg_filenames = [dummy_bg_files[0], dummy_bg_files[1]]
        mock_np_choice.side_effect = [chosen_fg_filenames, chosen_bg_filenames]

        # Mock the OverlaidSample instance and its __getitem__
        mock_sample_instance = MockOverlaidSample.return_value
        mock_sample_instance.__getitem__.return_value = (
            np.random.rand(*shape),  # input_image (overlaid)
            np.random.rand(*shape)   # output_image (e.g., foreground mask)
        )
        generator.reader = MagicMock(return_value=mock_sample_instance) # Mock reader

        # Act
        batch_x, batch_y = next(generator)

        # Assert
        mock_np_choice.assert_has_calls([
            call(generator.foreground_filenames, batch_size),
            call(generator.background_filenames, batch_size)
        ])
        
        self.assertEqual(generator.reader.call_count, batch_size)
        expected_reader_calls = []
        for i in range(batch_size):
            expected_reader_calls.append(call((chosen_fg_filenames[i], chosen_bg_filenames[i])))
        generator.reader.assert_has_calls(expected_reader_calls)
        
        self.assertEqual(mock_sample_instance.__getitem__.call_count, batch_size)

        self.assertEqual(batch_x.shape, (batch_size, *shape))
        self.assertEqual(batch_y.shape, (batch_size, *shape))

    def test_iter(self):
        # Arrange
        generator = OverlayGenerator(patterns=[], batch_size=4, shape=(128,128,3))
        # Act
        iterator_self = iter(generator)
        # Assert
        self.assertIs(iterator_self, generator)

if __name__ == '__main__':
    unittest.main()
