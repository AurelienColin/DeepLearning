import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from generators.normalizer import Normalizer
from generators.base_generators import BatchGenerator # For creating a dummy underlying generator


class TestNormalizer(unittest.TestCase):

    def setUp(self):
        # Common attributes
        self.channels = 3
        self.batch_size = 2
        self.height, self.width = 32, 32
        self.input_shape = (self.height, self.width, self.channels)
        self.output_shape_underlying = (self.height, self.width, self.channels) # Assuming same for simplicity

        # Mock underlying generator
        self.mock_underlying_generator = MagicMock(spec=BatchGenerator)
        self.mock_underlying_generator.input_shape = self.input_shape
        self.mock_underlying_generator.output_shape = self.output_shape_underlying # Normalizer normalizes outputs too
        self.mock_underlying_generator.batch_size = self.batch_size
        
        # Data for the underlying generator to return during __init__ of Normalizer
        # Let's make it simple: first half of iterations return all 1s, second half all 3s for inputs
        # Outputs can be zeros, as they are not used for calculating mean/std.
        self.init_iterations = Normalizer.ITERATIONS # Default is 100
        
        self.data_for_mean_calc = []
        for i in range(self.init_iterations):
            if i < self.init_iterations / 2:
                # Batch of ones
                inputs = np.ones((self.batch_size, *self.input_shape), dtype=np.float32)
            else:
                # Batch of threes
                inputs = np.full((self.batch_size, *self.input_shape), 3.0, dtype=np.float32)
            outputs = np.zeros((self.batch_size, *self.output_shape_underlying), dtype=np.float32)
            self.data_for_mean_calc.append((inputs, outputs))

        # Data for the __next__ call test (after initialization)
        self.test_input_data = np.full((self.batch_size, *self.input_shape), 2.0, dtype=np.float32)
        self.test_output_data_underlying = np.full((self.batch_size, *self.output_shape_underlying), 2.0, dtype=np.float32)
        self.data_for_next_call = (self.test_input_data, self.test_output_data_underlying)


    @patch('generators.normalizer.Normalizer.ITERATIONS', 2) # Reduce iterations for faster test
    def test_initialization_calculates_mean_std(self):
        # Arrange
        # Configure side_effect for the multiple calls to underlying_generator.__next__
        # during Normalizer's __init__
        # For ITERATIONS=2: 1 call for mean loop 1, 1 for mean loop 2, 
        # then 1 for std loop 1, 1 for std loop 2
        # Total 2*ITERATIONS calls.
        # We use simplified data: batch of 1s, then batch of 3s.
        # Expected mean: (1+3)/2 = 2.0 for all channels
        # Expected std: (|1-2| + |3-2|)/2 = (1+1)/2 = 1.0 for all channels
        
        # Data for mean calculation (first ITERATIONS calls)
        init_data_phase1 = [
            (np.ones((self.batch_size, *self.input_shape), dtype=np.float32), np.zeros((self.batch_size, *self.output_shape_underlying))),
            (np.full((self.batch_size, *self.input_shape), 3.0, dtype=np.float32), np.zeros((self.batch_size, *self.output_shape_underlying)))
        ]
        # Data for std calculation (next ITERATIONS calls) - uses the same input sequence
        init_data_phase2 = init_data_phase1[:]
        
        self.mock_underlying_generator.__next__.side_effect = init_data_phase1 + init_data_phase2

        # Act
        normalizer = Normalizer(channels=self.channels, generator=self.mock_underlying_generator)

        # Assert
        self.assertEqual(self.mock_underlying_generator.__next__.call_count, 2 * Normalizer.ITERATIONS)
        
        expected_means = np.full(self.channels, 2.0)
        expected_stds = np.full(self.channels, 1.0)
        
        np.testing.assert_array_almost_equal(normalizer.means, expected_means, decimal=6)
        np.testing.assert_array_almost_equal(normalizer.stds, expected_stds, decimal=6)

        # Check shapes are copied from underlying generator
        self.assertEqual(normalizer.input_shape, self.input_shape)
        self.assertEqual(normalizer.output_shape, self.output_shape_underlying) # Output shape is preserved

    @patch('generators.normalizer.Normalizer.ITERATIONS', 1) # Minimal iterations for this test
    def test_next_normalizes_data(self):
        # Arrange
        # For ITERATIONS=1:
        # Mean calc: 1 call. Let input be all 2s. Mean = 2.
        # Std calc: 1 call. Let input be all 2s. Std for |2-2| = 0. This is problematic for division.
        # Let's make std calc inputs different.
        # Mean calc: Input batch of all 1s. Mean = 1.
        # Std calc: Input batch of all 3s. Std for |3-1| = 2. Std = 2.
        
        mean_calc_data = (np.ones((self.batch_size, *self.input_shape), dtype=np.float32), np.zeros((self.batch_size, *self.output_shape_underlying)))
        std_calc_data = (np.full((self.batch_size, *self.input_shape), 3.0, dtype=np.float32), np.zeros((self.batch_size, *self.output_shape_underlying)))
        
        # Data for the actual __next__ call after initialization
        test_inputs = np.full((self.batch_size, *self.input_shape), 3.0, dtype=np.float32) # e.g. all 3s
        test_outputs_underlying = np.full((self.batch_size, *self.output_shape_underlying), 1.0, dtype=np.float32) # e.g. all 1s

        self.mock_underlying_generator.__next__.side_effect = [
            mean_calc_data, # For mean calculation loop in __init__
            std_calc_data,  # For std calculation loop in __init__
            (test_inputs.copy(), test_outputs_underlying.copy()) # For the actual next(normalizer) call
        ]
        
        normalizer = Normalizer(channels=self.channels, generator=self.mock_underlying_generator)
        
        # Expected mean = 1.0, std = 2.0
        # For test_inputs (all 3s): (3 - 1) / 2 = 1.0
        # For test_outputs_underlying (all 1s): (1 - 1) / 2 = 0.0
        expected_normalized_inputs = np.full((self.batch_size, *self.input_shape), 1.0, dtype=np.float32)
        expected_normalized_outputs = np.full((self.batch_size, *self.output_shape_underlying), 0.0, dtype=np.float32)

        # Act
        processed_inputs, processed_outputs = next(normalizer)

        # Assert
        # Check that underlying generator was called 2*ITERATIONS for init + 1 for the next() call
        self.assertEqual(self.mock_underlying_generator.__next__.call_count, 2 * Normalizer.ITERATIONS + 1)
        
        np.testing.assert_array_almost_equal(normalizer.means, np.full(self.channels, 1.0), decimal=6)
        np.testing.assert_array_almost_equal(normalizer.stds, np.full(self.channels, 2.0), decimal=6)
        
        np.testing.assert_array_almost_equal(processed_inputs, expected_normalized_inputs, decimal=6)
        np.testing.assert_array_almost_equal(processed_outputs, expected_normalized_outputs, decimal=6)
        
        # Ensure original shapes are maintained
        self.assertEqual(processed_inputs.shape, (self.batch_size, *self.input_shape))
        self.assertEqual(processed_outputs.shape, (self.batch_size, *self.output_shape_underlying))

    @patch('generators.normalizer.Normalizer.ITERATIONS', 1)
    def test_next_handles_zero_std(self):
        # Arrange
        # If std is zero, normalization should ideally not divide by zero.
        # The current implementation will divide by zero if std is zero.
        # This test highlights that. A more robust implementation might add a small epsilon to std.
        
        # Mean calc: Input batch of all 2s. Mean = 2.
        # Std calc: Input batch of all 2s. Std for |2-2| = 0. Std = 0.
        mean_calc_data = (np.full((self.batch_size, *self.input_shape), 2.0, dtype=np.float32), np.zeros((self.batch_size, *self.output_shape_underlying)))
        std_calc_data = (np.full((self.batch_size, *self.input_shape), 2.0, dtype=np.float32), np.zeros((self.batch_size, *self.output_shape_underlying)))
        
        test_inputs = np.full((self.batch_size, *self.input_shape), 2.0, dtype=np.float32)
        test_outputs_underlying = np.full((self.batch_size, *self.output_shape_underlying), 2.0, dtype=np.float32)

        self.mock_underlying_generator.__next__.side_effect = [
            mean_calc_data, std_calc_data, (test_inputs.copy(), test_outputs_underlying.copy())
        ]
        
        normalizer = Normalizer(channels=self.channels, generator=self.mock_underlying_generator)
        
        # Expect means = 2.0, stds = 0.0
        # (inputs - 2.0) / 0.0 will result in NaN or Inf.
        
        # Act
        processed_inputs, processed_outputs = next(normalizer)
        
        # Assert
        np.testing.assert_array_almost_equal(normalizer.means, np.full(self.channels, 2.0), decimal=6)
        np.testing.assert_array_almost_equal(normalizer.stds, np.full(self.channels, 0.0), decimal=6)

        # With std=0, (X - mean)/std results in NaN if X==mean, or +/-Inf if X!=mean
        # In this case, inputs and outputs are same as mean, so (2-2)/0 -> 0/0 -> NaN
        self.assertTrue(np.all(np.isnan(processed_inputs)))
        self.assertTrue(np.all(np.isnan(processed_outputs)))


if __name__ == '__main__':
    unittest.main()
