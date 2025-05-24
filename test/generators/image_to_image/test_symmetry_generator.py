import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_image.symmetry_generator import VerticalSymmetryGenerator
from generators.base_generators import BatchGenerator # For creating a dummy underlying generator


class TestVerticalSymmetryGenerator(unittest.TestCase):

    def test_initialization(self):
        # Arrange
        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        mock_underlying_generator.input_shape = (128, 128, 3)
        mock_underlying_generator.output_shape = (128, 128, 1) # Example: output could be a mask
        mock_underlying_generator.batch_size = 4
        
        # Act
        generator = VerticalSymmetryGenerator(generator=mock_underlying_generator)

        # Assert
        # Shapes and batch_size should be copied from the underlying generator
        self.assertEqual(generator.input_shape, mock_underlying_generator.input_shape)
        self.assertEqual(generator.output_shape, mock_underlying_generator.output_shape)
        self.assertEqual(generator.batch_size, mock_underlying_generator.batch_size)
        self.assertIs(generator.generator, mock_underlying_generator)

    @patch('numpy.random.randint')
    def test_next_horizontal_flip_logic(self, mock_randint):
        # Arrange
        batch_size = 4 # Must be > 1 to test both directions if possible
        height, width, channels_in = 64, 64, 3
        channels_out = 1 # Example for output (e.g. a mask)

        input_shape = (height, width, channels_in)
        output_shape = (height, width, channels_out)

        # Create mock input and output data for the underlying generator
        # Create easily identifiable data (e.g., sequential numbers)
        mock_inputs_batch = np.arange(batch_size * height * width * channels_in, dtype=np.float32).reshape(batch_size, *input_shape)
        mock_outputs_batch = np.arange(batch_size * height * width * channels_out, dtype=np.float32).reshape(batch_size, *output_shape)
        
        # Store originals for comparison
        original_mock_inputs_batch = mock_inputs_batch.copy()
        original_mock_outputs_batch = mock_outputs_batch.copy()

        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        mock_underlying_generator.input_shape = input_shape
        mock_underlying_generator.output_shape = output_shape
        mock_underlying_generator.batch_size = batch_size
        mock_underlying_generator.__next__.return_value = (mock_inputs_batch, mock_outputs_batch)

        generator = VerticalSymmetryGenerator(generator=mock_underlying_generator)
        
        # Control the random directions: 0 (flip), 0 (flip), 1 (no flip), 1 (no flip)
        # randint(0,1,size) * 2 - 1 results in -1 (flip) or 1 (no flip)
        # To get this, randint should return: 0, 0, 1, 1 for a batch of 4.
        # However, the code is `np.random.randint(0, 1, inputs.shape[0]) * 2 - 1`
        # This will always produce -1 because randint(0,1) is always 0. So all images are flipped.
        # Let's test this actual behavior.
        # If the intention was 50/50 flip, it should be randint(0,2,size).
        
        # For the current code: randint(0,1,...) always returns array of 0s.
        # So directions will be all -1s (all flipped).
        mock_randint.return_value = np.array([0] * batch_size) # This makes directions = [-1]*batch_size
        
        # Act
        processed_inputs, processed_outputs = next(generator)

        # Assert
        mock_underlying_generator.__next__.assert_called_once()
        mock_randint.assert_called_once_with(0, 1, batch_size)

        # Check shapes
        self.assertEqual(processed_inputs.shape, (batch_size, *input_shape))
        self.assertEqual(processed_outputs.shape, (batch_size, *output_shape))

        # Verify flipping logic (all should be flipped based on current code)
        for i in range(batch_size):
            np.testing.assert_array_equal(processed_inputs[i], original_mock_inputs_batch[i, :, ::-1])
            if processed_outputs.ndim == processed_inputs.ndim: # Check if output was also flipped
                 np.testing.assert_array_equal(processed_outputs[i], original_mock_outputs_batch[i, :, ::-1])
    
    @patch('numpy.random.randint')
    def test_next_horizontal_flip_logic_mixed_directions(self, mock_randint):
        # This test assumes the np.random.randint was intended to be np.random.randint(0, 2, size)
        # to allow for mixed directions. We will patch it to behave this way.
        # Arrange
        batch_size = 2
        height, width, channels_in = 4, 4, 1 # Smaller for easier manual verification
        input_shape = (height, width, channels_in)
        output_shape = input_shape # Assuming output is also an image of same dims

        # Create simple, identifiable data
        img1_in = np.array([[[1],[2],[3],[4]], [[5],[6],[7],[8]], [[9],[10],[11],[12]], [[13],[14],[15],[16]]], dtype=np.float32).reshape(input_shape)
        img2_in = img1_in * 10
        mock_inputs_batch = np.stack([img1_in, img2_in])

        img1_out = img1_in * 0.1
        img2_out = img2_in * 0.1
        mock_outputs_batch = np.stack([img1_out, img2_out])
        
        original_mock_inputs_batch = mock_inputs_batch.copy()
        original_mock_outputs_batch = mock_outputs_batch.copy()

        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        mock_underlying_generator.input_shape = input_shape
        mock_underlying_generator.output_shape = output_shape
        mock_underlying_generator.batch_size = batch_size
        mock_underlying_generator.__next__.return_value = (mock_inputs_batch, mock_outputs_batch)

        generator = VerticalSymmetryGenerator(generator=mock_underlying_generator)
        
        # Simulate randint(0,2,size) behavior:
        # First image: direction = 0 -> mapped to -1 (flip)
        # Second image: direction = 1 -> mapped to 1 (no flip)
        mock_randint.return_value = np.array([0, 1]) # This makes directions = [-1, 1]
        
        # Act
        processed_inputs, processed_outputs = next(generator)

        # Assert
        # Verify inputs
        np.testing.assert_array_equal(processed_inputs[0], original_mock_inputs_batch[0, :, ::-1]) # Flipped
        np.testing.assert_array_equal(processed_inputs[1], original_mock_inputs_batch[1, :, ::1])  # Not flipped
        
        # Verify outputs (should follow same transformations)
        np.testing.assert_array_equal(processed_outputs[0], original_mock_outputs_batch[0, :, ::-1]) # Flipped
        np.testing.assert_array_equal(processed_outputs[1], original_mock_outputs_batch[1, :, ::1])  # Not flipped

    def test_next_output_ndim_mismatch(self):
        # Test case where output.ndim != input.ndim, so output should not be flipped.
        # Arrange
        batch_size = 1
        input_shape = (4, 4, 3)
        output_shape_scalar = (1,) # e.g. a classification label, ndim is 1 (after batch)

        mock_inputs_batch = np.random.rand(batch_size, *input_shape).astype(np.float32)
        # Output is (batch_size, 1) e.g. [[0], [1]]
        mock_outputs_batch = np.random.randint(0, 2, (batch_size, *output_shape_scalar)).astype(np.float32)
        
        original_mock_inputs_batch = mock_inputs_batch.copy()
        original_mock_outputs_batch = mock_outputs_batch.copy()

        mock_underlying_generator = MagicMock(spec=BatchGenerator)
        mock_underlying_generator.input_shape = input_shape
        mock_underlying_generator.output_shape = output_shape_scalar
        mock_underlying_generator.batch_size = batch_size
        mock_underlying_generator.__next__.return_value = (mock_inputs_batch, mock_outputs_batch)

        generator = VerticalSymmetryGenerator(generator=mock_underlying_generator)
        
        # Act
        processed_inputs, processed_outputs = next(generator)

        # Assert
        # Input should be flipped (default behavior of randint(0,1,size) leads to flip)
        np.testing.assert_array_equal(processed_inputs[0], original_mock_inputs_batch[0, :, ::-1])
        # Output should NOT be flipped because ndim is different
        np.testing.assert_array_equal(processed_outputs, original_mock_outputs_batch)


if __name__ == '__main__':
    unittest.main()
