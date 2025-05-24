import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_image.grey_in_color_out_generator import GreyInColorOutGenerator
from generators.base_generators import BatchGenerator # For creating a dummy underlying generator


class TestGreyInColorOutGenerator(unittest.TestCase):

    def test_initialization(self):
        # Arrange
        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        mock_underlying_generator.input_shape = (128, 128, 3)
        mock_underlying_generator.output_shape = (128, 128, 3)
        mock_underlying_generator.batch_size = 4
        
        # Act
        # The GreyInColorOutGenerator's __init__ just calls super().__init__
        # which copies attributes from the underlying generator
        generator = GreyInColorOutGenerator(generator=mock_underlying_generator)

        # Assert
        # Check that properties from the underlying generator are copied
        self.assertEqual(generator.input_shape, (128, 128, 1)) # Input becomes greyscale
        self.assertEqual(generator.output_shape, (128, 128, 3)) # Output remains color
        self.assertEqual(generator.batch_size, 4)
        self.assertIs(generator.generator, mock_underlying_generator)

    def test_next_greyscale_conversion(self):
        # Arrange
        batch_size = 2
        input_shape_color = (64, 64, 3)
        output_shape_color = (64, 64, 3) # Could be different, but using same for simplicity

        # Create mock input and output data for the underlying generator
        # Input images are color
        mock_input_batch = np.random.rand(batch_size, *input_shape_color).astype(np.float32)
        # Output images are also color (as per typical autoencoder/colorization setup)
        mock_output_batch = np.random.rand(batch_size, *output_shape_color).astype(np.float32)

        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        # Set shapes for the underlying generator
        mock_underlying_generator.input_shape = input_shape_color
        mock_underlying_generator.output_shape = output_shape_color # output shape is preserved
        mock_underlying_generator.batch_size = batch_size
        
        # Configure the __next__ method of the mock underlying generator
        mock_underlying_generator.__next__.return_value = (mock_input_batch, mock_output_batch)

        # Initialize the GreyInColorOutGenerator
        # Its input_shape should be (H, W, 1) and output_shape (H, W, 3)
        generator = GreyInColorOutGenerator(generator=mock_underlying_generator)
        
        # Expected input shape after greyscale conversion
        expected_input_shape_grey = (batch_size, input_shape_color[0], input_shape_color[1], 1)

        # Act
        processed_inputs, processed_outputs = next(generator)

        # Assert
        # Check that the underlying generator's __next__ was called
        mock_underlying_generator.__next__.assert_called_once()

        # Check the shapes of the processed batches
        self.assertEqual(processed_inputs.shape, expected_input_shape_grey)
        self.assertEqual(processed_outputs.shape, (batch_size, *output_shape_color))

        # Check the conversion logic (mean across the color channel)
        # Take the first image from the batch for detailed check
        expected_greyscale_image = np.mean(mock_input_batch[0], axis=2, keepdims=True)
        np.testing.assert_array_almost_equal(processed_inputs[0], expected_greyscale_image, decimal=6)
        
        # Ensure output is untouched
        np.testing.assert_array_equal(processed_outputs, mock_output_batch)


if __name__ == '__main__':
    unittest.main()
