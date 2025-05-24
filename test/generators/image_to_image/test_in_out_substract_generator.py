import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_image.in_out_substract_generator import InOutSubstractGenerator
from generators.base_generators import BatchGenerator # For creating a dummy underlying generator


class TestInOutSubstractGenerator(unittest.TestCase):

    def test_initialization(self):
        # Arrange
        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        mock_underlying_generator.input_shape = (128, 128, 3) # Input is color
        mock_underlying_generator.output_shape = (128, 128, 3) # Output from underlying is also color
        mock_underlying_generator.batch_size = 4
        
        # Act
        generator = InOutSubstractGenerator(generator=mock_underlying_generator)

        # Assert
        # Input shape remains the same as the underlying generator's input
        self.assertEqual(generator.input_shape, (128, 128, 3)) 
        # Output shape becomes single channel (difference map)
        self.assertEqual(generator.output_shape, (128, 128, 1)) 
        self.assertEqual(generator.batch_size, 4)
        self.assertIs(generator.generator, mock_underlying_generator)

    def test_next_difference_calculation(self):
        # Arrange
        batch_size = 2
        height, width = 64, 64
        input_shape_color = (height, width, 3)
        # Underlying generator's output is also color
        output_shape_color_underlying = (height, width, 3) 

        # Create mock input and output data for the underlying generator
        mock_inputs_batch = np.random.rand(batch_size, *input_shape_color).astype(np.float32)
        # Outputs from underlying generator (e.g. original image and its reconstruction)
        mock_outputs_batch_underlying = np.random.rand(batch_size, *output_shape_color_underlying).astype(np.float32)

        # Ensure some differences for the test
        # Make one pixel significantly different in the first image pair
        mock_outputs_batch_underlying[0, 0, 0, :] = mock_inputs_batch[0, 0, 0, :] + 0.5 
        # Make another pixel very similar (small difference)
        mock_outputs_batch_underlying[0, 1, 1, :] = mock_inputs_batch[0, 1, 1, :] + 0.05


        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        mock_underlying_generator.input_shape = input_shape_color
        mock_underlying_generator.output_shape = output_shape_color_underlying
        mock_underlying_generator.batch_size = batch_size
        mock_underlying_generator.__next__.return_value = (mock_inputs_batch, mock_outputs_batch_underlying)

        generator = InOutSubstractGenerator(generator=mock_underlying_generator)
        
        # Expected output shape (batch_size, H, W, 1)
        expected_output_shape_diff = (batch_size, height, width, 1)

        # Act
        processed_inputs, processed_outputs_diff = next(generator)

        # Assert
        mock_underlying_generator.__next__.assert_called_once()

        # Check shapes
        self.assertEqual(processed_inputs.shape, (batch_size, *input_shape_color))
        self.assertEqual(processed_outputs_diff.shape, expected_output_shape_diff)

        # Check input passthrough
        np.testing.assert_array_equal(processed_inputs, mock_inputs_batch)

        # Check difference calculation for the first image
        input_mono_0 = np.mean(mock_inputs_batch[0], axis=-1)
        output_mono_0 = np.mean(mock_outputs_batch_underlying[0], axis=-1)
        diff_0 = np.abs(output_mono_0 - input_mono_0)
        expected_diff_image_0 = np.where(diff_0 > 0.2, 1., 0.)[:, :, None]
        
        np.testing.assert_array_almost_equal(processed_outputs_diff[0], expected_diff_image_0, decimal=6)
        
        # Specifically check the test pixels in the first image
        # Significant difference pixel (0,0)
        self.assertEqual(processed_outputs_diff[0, 0, 0, 0], 1.0) 
        # Small difference pixel (1,1)
        self.assertEqual(processed_outputs_diff[0, 1, 1, 0], 0.0)


if __name__ == '__main__':
    unittest.main()
