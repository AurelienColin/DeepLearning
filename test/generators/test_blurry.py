import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from generators.blurry import BlurryGenerator
from generators.base_generators import BatchGenerator # For creating a dummy underlying generator


class TestBlurryGenerator(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.batch_size = 2
        self.height, self.width = 32, 32
        self.channels_in = 3
        self.channels_out_underlying = 3 # Output channels from the underlying generator

        self.input_shape_underlying = (self.height, self.width, self.channels_in)
        self.output_shape_underlying = (self.height, self.width, self.channels_out_underlying)

        # Mock underlying generator
        self.mock_underlying_generator = MagicMock(spec=BatchGenerator)
        self.mock_underlying_generator.input_shape = self.input_shape_underlying
        self.mock_underlying_generator.output_shape = self.output_shape_underlying
        self.mock_underlying_generator.batch_size = self.batch_size
        
        # Create dummy batch data
        self.mock_inputs_batch = np.random.rand(self.batch_size, *self.input_shape_underlying).astype(np.float32)
        self.mock_outputs_batch_underlying = np.random.rand(self.batch_size, *self.output_shape_underlying).astype(np.float32)
        
        self.mock_underlying_generator.__next__.return_value = (
            self.mock_inputs_batch.copy(), # Use copies to avoid in-place modification issues
            self.mock_outputs_batch_underlying.copy()
        )

        self.blurry_generator = BlurryGenerator(generator=self.mock_underlying_generator)

    def test_initialization(self):
        # Assert
        # Input shape remains the same
        self.assertEqual(self.blurry_generator.input_shape, self.input_shape_underlying)
        # Output shape has an added channel for sigma values
        expected_output_shape = (self.height, self.width, self.channels_out_underlying + 1)
        self.assertEqual(self.blurry_generator.output_shape, expected_output_shape)
        self.assertEqual(self.blurry_generator.batch_size, self.batch_size)
        self.assertIs(self.blurry_generator.generator, self.mock_underlying_generator)
        self.assertEqual(self.blurry_generator.SIGMAX_MAX, 1)

    @patch('scipy.ndimage.gaussian_filter')
    def test_blur_method(self, mock_gaussian_filter):
        # Arrange
        test_array = np.random.rand(10, 10, 3).astype(np.float32)
        sigma_val = 0.5
        # Mock gaussian_filter to return the input array to isolate clip behavior if needed,
        # or a specific known output. For this test, just returning input is fine.
        mock_gaussian_filter.return_value = test_array.copy() 

        # Act
        blurred_array = self.blurry_generator.blur(test_array, sigma_val)

        # Assert
        mock_gaussian_filter.assert_called_once_with(test_array, sigma_val, axes=(0,1))
        # np.clip should ensure values are within original min/max. Since filter is mocked to return original,
        # this test mainly ensures the call structure.
        np.testing.assert_array_equal(blurred_array, test_array)


    @patch('numpy.random.exponential')
    @patch('scipy.ndimage.gaussian_filter') # Mock the actual filter call within blur
    def test_next_data_transformation_and_shapes(self, mock_gaussian_filter, mock_np_exponential):
        # Arrange
        # Fixed sigma values for predictable testing
        test_sigmas = np.array([0.5, 0.8], dtype=np.float32) # One sigma per item in batch
        mock_np_exponential.return_value = test_sigmas

        # Mock gaussian_filter to simply return the input array,
        # so we can focus on testing the structure and concatenation.
        def filter_side_effect(array, sigma, axes):
            return array.copy() 
        mock_gaussian_filter.side_effect = filter_side_effect

        # Act
        processed_inputs, processed_outputs = next(self.blurry_generator)

        # Assert
        # Underlying generator called
        self.mock_underlying_generator.__next__.assert_called_once()
        # np.random.exponential called
        mock_np_exponential.assert_called_once_with(self.blurry_generator.SIGMAX_MAX, size=self.batch_size)

        # Gaussian filter call count: once per input image, once per output image in the batch
        self.assertEqual(mock_gaussian_filter.call_count, self.batch_size * 2) 
        
        # Check shapes
        self.assertEqual(processed_inputs.shape, (self.batch_size, *self.input_shape_underlying))
        expected_output_final_shape = (self.batch_size, self.height, self.width, self.channels_out_underlying + 1)
        self.assertEqual(processed_outputs.shape, expected_output_final_shape)

        # Check input processing (should be "blurred" - mocked to be same as original here)
        np.testing.assert_array_equal(processed_inputs, self.mock_inputs_batch)

        # Check output processing
        # Part 1: "blurred" underlying outputs
        np.testing.assert_array_equal(processed_outputs[..., :-1], self.mock_outputs_batch_underlying)

        # Part 2: appended sigma values
        # sigmas = np.clip(sigmas/4, 0, 1)
        expected_sigmas_processed = np.clip(test_sigmas / 4.0, 0, 1)
        # sigmas = sigmas[:, np.newaxis, np.newaxis, np.newaxis]
        # sigmas = np.tile(sigmas, (1, *inputs.shape[1:3], 1))
        # For each item in batch, the sigma value should be tiled across H, W dimensions
        for i in range(self.batch_size):
            expected_sigma_channel = np.full((self.height, self.width, 1), expected_sigmas_processed[i])
            np.testing.assert_array_almost_equal(processed_outputs[i, ..., -1:], expected_sigma_channel)


if __name__ == '__main__':
    unittest.main()
