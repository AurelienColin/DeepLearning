import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_tag.classification_generator import ClassificationGenerator
from samples.image_to_tag.image_to_tag_sample import ImageToTagSample
from output_spaces.output_space import OutputSpace # For type hinting mocks
from output_spaces.space_from_filesystem import CategorizationSpace
from output_spaces.space_from_json import TaggerSpace


class TestClassificationGenerator(unittest.TestCase):

    def common_init_setup(self, mock_tagger_space, mock_categorization_space, num_files, enforced_tags=None):
        shape = (128, 128, 3)
        batch_size = 4
        filenames = [f"file{i}.jpg" for i in range(num_files)]
        
        # Mock the output space instances
        mock_space_instance = MagicMock(spec=OutputSpace)
        mock_space_instance.filenames = filenames # Crucial for the generator's self.filenames
        mock_space_instance.output_shape = (10,) # Example output shape (e.g. 10 classes)

        if num_files > 1:
            mock_categorization_space.return_value = mock_space_instance
        else:
            mock_tagger_space.return_value = mock_space_instance

        generator = ClassificationGenerator(
            shape=shape, 
            batch_size=batch_size, 
            samples=filenames, # samples are filenames for this generator
            enforced_tag_names=enforced_tags
        )
        return generator, mock_space_instance, filenames, shape, batch_size

    @patch('output_spaces.space_from_json.TaggerSpace')
    @patch('output_spaces.space_from_filesystem.CategorizationSpace')
    def test_initialization_with_tagger_space(self, mock_categorization_space, mock_tagger_space):
        # Arrange (1 file triggers TaggerSpace)
        num_files = 1
        enforced_tags = ["tagA", "tagB"]
        generator, mock_space_instance, filenames, shape, batch_size = self.common_init_setup(
            mock_tagger_space, mock_categorization_space, num_files, enforced_tags
        )

        # Assert
        mock_tagger_space.assert_called_once_with(filenames, enforced_tag_names=enforced_tags)
        mock_categorization_space.assert_not_called()
        self.assertIs(generator.output_space, mock_space_instance)
        self.assertEqual(generator.filenames, filenames) # filenames from output_space
        self.assertEqual(generator.shape, shape)
        self.assertEqual(generator.batch_size, batch_size)
        self.assertEqual(generator.samples, filenames) # samples from init
        self.assertEqual(generator.input_shape, shape) # from BatchGenerator
        self.assertEqual(generator.output_shape, mock_space_instance.output_shape) # from output_space
        self.assertEqual(generator.enforced_tag_names, enforced_tags)


    @patch('output_spaces.space_from_json.TaggerSpace')
    @patch('output_spaces.space_from_filesystem.CategorizationSpace')
    def test_initialization_with_categorization_space(self, mock_categorization_space, mock_tagger_space):
        # Arrange (multiple files trigger CategorizationSpace)
        num_files = 5
        generator, mock_space_instance, filenames, shape, batch_size = self.common_init_setup(
            mock_tagger_space, mock_categorization_space, num_files, None # No enforced tags
        )

        # Assert
        mock_categorization_space.assert_called_once_with(filenames, enforced_tag_names=None)
        mock_tagger_space.assert_not_called()
        self.assertIs(generator.output_space, mock_space_instance)
        self.assertEqual(generator.filenames, filenames)
        self.assertEqual(generator.shape, shape)
        self.assertEqual(generator.batch_size, batch_size)
        self.assertEqual(generator.samples, filenames)
        self.assertEqual(generator.input_shape, shape)
        self.assertEqual(generator.output_shape, mock_space_instance.output_shape)


    @patch('samples.image_to_tag.image_to_tag_sample.ImageToTagSample')
    @patch('output_spaces.space_from_json.TaggerSpace') # Mock for init
    @patch('output_spaces.space_from_filesystem.CategorizationSpace') # Mock for init
    def test_reader(self, mock_cat_space, mock_tag_space, MockImageToTagSample):
        # Arrange
        # Initialize generator (using TaggerSpace for simplicity in this test)
        generator, mock_output_space, filenames, shape, _ = self.common_init_setup(
            mock_tag_space, mock_cat_space, 1 
        )
        mock_sample_instance = MockImageToTagSample.return_value
        test_filename = "dummy.jpg"

        # Act
        result = generator.reader(test_filename)

        # Assert
        MockImageToTagSample.assert_called_once_with(
            output_space=mock_output_space,
            input_filename=test_filename,
            shape=shape
        )
        self.assertEqual(result, mock_sample_instance)


    @patch('samples.image_to_tag.image_to_tag_sample.ImageToTagSample')
    @patch('output_spaces.space_from_json.TaggerSpace') # Mock for init
    @patch('output_spaces.space_from_filesystem.CategorizationSpace') # Mock for init
    def test_getitem_output_shapes(self, mock_cat_space, mock_tag_space, MockImageToTagSample):
        # Arrange
        num_files = 4 # Must be a multiple of batch_size for this simple test
        batch_size = 2
        input_img_shape = (64, 64, 3)
        output_vector_shape = (5,) # e.g. 5 classes/tags

        # Setup generator instance
        generator, mock_output_space, filenames, _, _ = self.common_init_setup(
            mock_tag_space, mock_cat_space, num_files
        )
        # Override parts for this specific test
        generator.shape = input_img_shape
        generator.batch_size = batch_size
        generator.output_space.output_shape = output_vector_shape
        generator.filenames = filenames[:num_files] # Ensure samples list matches num_files
        generator.samples = filenames[:num_files]


        # Mock the ImageToTagSample instance and its __getitem__
        mock_sample_instance = MockImageToTagSample.return_value
        mock_sample_instance.__getitem__.return_value = (
            np.random.rand(*input_img_shape),  # input_image
            np.random.randint(0, 2, size=output_vector_shape)   # output_tags (binary vector)
        )
        # Ensure the reader method returns our mocked sample
        generator.reader = MagicMock(return_value=mock_sample_instance)

        # Act
        batch_x, batch_y = generator.__getitem__(0) # Get the first batch

        # Assert
        self.assertEqual(batch_x.shape, (batch_size, *input_img_shape))
        self.assertEqual(batch_y.shape, (batch_size, *output_vector_shape))
        
        self.assertEqual(generator.reader.call_count, batch_size)
        for i in range(batch_size):
            # generator.filenames contains the list of files from the output_space,
            # which __getitem__ uses.
            generator.reader.assert_any_call(generator.filenames[i])
        
        self.assertEqual(mock_sample_instance.__getitem__.call_count, batch_size)


if __name__ == '__main__':
    unittest.main()
