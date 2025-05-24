import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

from generators.image_to_tag.custom.rating_as_float_generator import RatingAsFloatGenerator
from samples.image_to_tag.image_to_tag_sample import ImageToTagSample # Parent class uses this
from output_spaces.custom.rating_as_float_space import RatingAsFloatSpace


class TestRatingAsFloatGenerator(unittest.TestCase):

    @patch('output_spaces.custom.rating_as_float_space.RatingAsFloatSpace')
    def test_initialization_and_set_output_space(self, MockRatingAsFloatSpace):
        # Arrange
        shape = (128, 128, 3)
        batch_size = 4
        # Filenames are used by RatingAsFloatSpace to load data (e.g. from JSON files)
        # For this test, we just need a list of strings that will be passed to it.
        sample_filenames = ["data1.json", "data2.json"] 
        enforced_tags = ["rating"]

        mock_space_instance = MockRatingAsFloatSpace.return_value
        # RatingAsFloatSpace needs to provide a `filenames` attribute after it's initialized,
        # which the generator then copies to its own `self.filenames`.
        # This usually contains image file paths extracted from the JSON.
        mock_space_instance.filenames = ["img1.jpg", "img2.jpg", "img3.jpg"] 
        mock_space_instance.output_shape = (1,) # Example: a single float rating

        # Act
        generator = RatingAsFloatGenerator(
            shape=shape,
            batch_size=batch_size,
            samples=sample_filenames, # These are passed to constructor and then to RatingAsFloatSpace
            enforced_tag_names=enforced_tags
        )

        # Assert
        # Check that RatingAsFloatSpace was called correctly
        MockRatingAsFloatSpace.assert_called_once_with(sample_filenames, enforced_tag_names=enforced_tags)
        
        # Check that the generator's output_space is the mocked instance
        self.assertIs(generator.output_space, mock_space_instance)
        
        # Check that generator attributes are set correctly
        self.assertEqual(generator.shape, shape)
        self.assertEqual(generator.batch_size, batch_size)
        # The generator's `samples` attribute is what was passed during __init__
        self.assertEqual(generator.samples, sample_filenames) 
        # The generator's `filenames` attribute is copied from the `output_space` instance
        self.assertEqual(generator.filenames, mock_space_instance.filenames)
        self.assertEqual(generator.input_shape, shape) # Inherited from BatchGenerator
        self.assertEqual(generator.output_shape, mock_space_instance.output_shape) # From output_space
        self.assertEqual(generator.enforced_tag_names, enforced_tags)

    # Test for the 'reader' method is inherited from ClassificationGenerator.
    # We can do a simple check to ensure it's callable and uses the correct sample type if needed,
    # but the core logic is in the parent. Here, we verify it uses the output_space correctly.
    @patch('samples.image_to_tag.image_to_tag_sample.ImageToTagSample')
    @patch('output_spaces.custom.rating_as_float_space.RatingAsFloatSpace') # Mock for init
    def test_reader_uses_correct_output_space(self, MockRatingAsFloatSpace, MockImageToTagSample):
        # Arrange
        shape = (64, 64, 1)
        batch_size = 1
        sample_files = ["test_data.json"] # File for RatingAsFloatSpace
        
        mock_space_instance = MockRatingAsFloatSpace.return_value
        mock_space_instance.filenames = ["image_from_json.jpg"] # Image file derived by space
        mock_space_instance.output_shape = (1,)

        generator = RatingAsFloatGenerator(
            shape=shape, batch_size=batch_size, samples=sample_files
        )
        
        mock_sample_obj = MockImageToTagSample.return_value
        test_input_image_filename = "image_from_json.jpg" # This would be one of generator.filenames

        # Act
        result = generator.reader(test_input_image_filename)

        # Assert
        MockImageToTagSample.assert_called_once_with(
            output_space=mock_space_instance, # Crucially, this should be the RatingAsFloatSpace instance
            input_filename=test_input_image_filename,
            shape=shape
        )
        self.assertIs(result, mock_sample_obj)

    # Test for __getitem__ is also largely inherited.
    # We focus on ensuring that the interaction with a mocked reader (which uses the output_space)
    # and the resulting batch shapes are correct.
    @patch('samples.image_to_tag.image_to_tag_sample.ImageToTagSample')
    @patch('output_spaces.custom.rating_as_float_space.RatingAsFloatSpace') # Mock for init
    def test_getitem_output_shapes(self, MockRatingAsFloatSpace, MockImageToTagSample):
        # Arrange
        shape = (32, 32, 3)
        batch_size = 2
        num_images_from_space = 4 # e.g. RatingAsFloatSpace found 4 images
        output_vector_shape = (1,) # single float rating

        sample_files_for_space = ["data.json"] # Input to RatingAsFloatGenerator

        mock_space_instance = MockRatingAsFloatSpace.return_value
        # These are the image file paths that the space makes available
        mock_space_instance.filenames = [f"img_{i}.png" for i in range(num_images_from_space)]
        mock_space_instance.output_shape = output_vector_shape

        generator = RatingAsFloatGenerator(
            shape=shape, batch_size=batch_size, samples=sample_files_for_space
        )
        # Ensure the generator's samples list for __getitem__ is what the space provides
        # The generator's `samples` attribute remains `sample_files_for_space`.
        # `BatchGenerator.__getitem__` iterates `self.samples_files_list` which is `self.filenames`
        # if `self.filenames_are_samples` is true (default).
        # `ClassificationGenerator` sets `self.filenames = self.output_space.filenames`.

        mock_sample_instance = MockImageToTagSample.return_value
        mock_sample_instance.__getitem__.return_value = (
            np.random.rand(*shape),
            np.random.rand(*output_vector_shape).astype(np.float32)
        )
        generator.reader = MagicMock(return_value=mock_sample_instance)

        # Act
        batch_x, batch_y = generator.__getitem__(0) # Get the first batch

        # Assert
        self.assertEqual(batch_x.shape, (batch_size, *shape))
        self.assertEqual(batch_y.shape, (batch_size, *output_vector_shape))

        self.assertEqual(generator.reader.call_count, batch_size)
        # __getitem__ uses self.filenames (which comes from output_space.filenames)
        for i in range(batch_size):
            generator.reader.assert_any_call(mock_space_instance.filenames[i])
        
        self.assertEqual(mock_sample_instance.__getitem__.call_count, batch_size)


if __name__ == '__main__':
    unittest.main()
