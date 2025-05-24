import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from generators.image_to_image.autoencoder_generator import AutoEncoderGenerator
from samples.image_to_image.autoencoder_sample import AutoEncodingSample


class TestAutoEncoderGenerator(unittest.TestCase):

    @patch('samples.image_to_image.autoencoder_sample.AutoEncodingSample')
    def test_reader(self, MockAutoEncodingSample):
        # Arrange
        mock_sample_instance = MockAutoEncodingSample.return_value
        generator = AutoEncoderGenerator(shape=(128, 128, 3), batch_size=4, samples=[])
        test_filename = "dummy.jpg"

        # Act
        result = generator.reader(test_filename)

        # Assert
        MockAutoEncodingSample.assert_called_once_with(test_filename, (128, 128, 3))
        self.assertEqual(result, mock_sample_instance)

    def test_initialization(self):
        # Arrange
        shape = (256, 256, 3)
        batch_size = 16
        samples = ["sample1.png", "sample2.png"]

        # Act
        generator = AutoEncoderGenerator(shape=shape, batch_size=batch_size, samples=samples)

        # Assert
        self.assertEqual(generator.shape, shape)
        self.assertEqual(generator.batch_size, batch_size)
        self.assertEqual(generator.samples, samples)
        self.assertEqual(generator.input_shape, shape)
        self.assertEqual(generator.output_shape, shape)

    @patch('samples.image_to_image.autoencoder_sample.AutoEncodingSample')
    def test_getitem_output_shapes(self, MockAutoEncodingSample):
        # Arrange
        shape = (64, 64, 1)
        batch_size = 2
        num_samples = 4 # Should be a multiple of batch_size for this test
        samples = [f"dummy_{i}.png" for i in range(num_samples)]

        # Mock the behavior of AutoEncodingSample's __getitem__
        # to return appropriately shaped numpy arrays
        mock_sample_instance = MockAutoEncodingSample.return_value
        mock_sample_instance.__getitem__.return_value = (
            np.random.rand(*shape),  # input_image
            np.random.rand(*shape)   # output_image
        )

        generator = AutoEncoderGenerator(shape=shape, batch_size=batch_size, samples=samples)
        generator.reader = MagicMock(return_value=mock_sample_instance) # Mock reader to return the mocked sample

        # Act
        batch_x, batch_y = generator.__getitem__(0) # Get the first batch

        # Assert
        self.assertEqual(batch_x.shape, (batch_size, *shape))
        self.assertEqual(batch_y.shape, (batch_size, *shape))
        
        # Check that reader was called for each sample in the batch
        self.assertEqual(generator.reader.call_count, batch_size)
        for i in range(batch_size):
            generator.reader.assert_any_call(samples[i])


if __name__ == '__main__':
    unittest.main()
