import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import typing

from src.generators.base_generators import BatchGenerator
from src.samples.sample import Sample

class TestBatchGenerator(unittest.TestCase):

    def test_instantiation(self) -> None:
        filenames: typing.List[str] = ["file1.txt", "file2.txt"]
        batch_size: int = 2
        shape: typing.Tuple[int, int, int] = (64, 64, 3)
        generator = BatchGenerator(filenames=filenames, batch_size=batch_size, shape=shape)
        self.assertIsInstance(generator, BatchGenerator)
        self.assertEqual(generator.batch_size, batch_size)
        self.assertEqual(generator.shape, shape)
        self.assertTrue(np.array_equal(generator.filenames, np.array(filenames)))

    @patch('src.generators.base_generators.ThreadPool')
    def test_iteration_and_batch_processing_success(self, mock_thread_pool_class: MagicMock) -> None:
        filenames: typing.List[str] = ["file1.img", "file2.img", "file3.img", "file4.img"]
        batch_size: int = 2
        height, width, channels = 64, 64, 3
        shape: typing.Tuple[int, int, int] = (height, width, channels)

        # Configure the mock ThreadPool and its map method
        mock_pool_instance = MagicMock()
        mock_thread_pool_class.return_value.__enter__.return_value = mock_pool_instance

        # Create MagicMock objects that simulate Sample instances
        # These will be returned by the mocked pool.map
        mock_sample1 = MagicMock(spec=Sample)
        mock_sample1.input_data = np.random.rand(height, width, channels)
        mock_sample1.output_data = np.random.rand(height, width, channels)
        
        mock_sample2 = MagicMock(spec=Sample)
        mock_sample2.input_data = np.random.rand(height, width, channels)
        mock_sample2.output_data = np.random.rand(height, width, channels)
        
        # This is what the mocked ThreadPool.map will return
        # mock_pool_instance.map.return_value = [mock_sample1, mock_sample2] # Removed

        generator = BatchGenerator(filenames=filenames, batch_size=batch_size, shape=shape)
        
        # The generator.reader is called by pool.map.
        # We need to mock it for call_count verification and to ensure it's callable.
        def actual_reader_mock_side_effect(filename: str) -> Sample:
            # This function will serve as the mock for generator.reader
            # It should return a mock Sample object with the necessary data attributes
            s_mock = MagicMock(spec=Sample)
            if filename == "file1.img": # Or some other way to distinguish if needed for data
                s_mock.input_data = mock_sample1.input_data
                s_mock.output_data = mock_sample1.output_data
            else: # For file2.img, file3.img, file4.img etc.
                s_mock.input_data = mock_sample2.input_data # Simplification for test
                s_mock.output_data = mock_sample2.output_data
            return s_mock

        generator.reader = MagicMock(side_effect=actual_reader_mock_side_effect)

        # Configure mock_pool_instance.map to call generator.reader for each filename
        # The actual filenames passed to map will be chosen by np.random.choice in BatchGenerator
        # So, the side_effect for map should take a function and an iterable, and apply the function to the iterable.
        def map_side_effect(func_to_call_in_map, filenames_for_map):
            # func_to_call_in_map will be generator.reader
            # filenames_for_map will be the list of filenames chosen by np.random.choice
            results = []
            for fname in filenames_for_map:
                results.append(func_to_call_in_map(fname))
            return results
        
        mock_pool_instance.map.side_effect = map_side_effect
        
        # Mock np.random.choice to control which files are selected, ensuring consistency
        # This makes sure that the filenames passed to generator.reader (via map_side_effect)
        # are predictable, e.g., "file1.img" and "file2.img" if batch_size is 2.
        # Otherwise, the data in mock_sample1/mock_sample2 might not align with the files processed.
        with patch('numpy.random.choice', return_value=filenames[:batch_size]) as mock_np_choice:
            inputs, outputs = next(generator)
            # Ensure np.random.choice was called with positional arg for size
            mock_np_choice.assert_called_once_with(generator.filenames, generator.batch_size, replace=False)

        self.assertIsInstance(inputs, np.ndarray)
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(inputs.shape, (batch_size, height, width, channels))
        self.assertEqual(outputs.shape, (batch_size, height, width, channels))
        
        # Check that ThreadPool was used
        mock_thread_pool_class.assert_called_once_with(processes=batch_size)
        mock_pool_instance.map.assert_called_once()
        
        # Check that reader was called for the files selected by np.random.choice
        # This is a bit tricky because np.random.choice is involved.
        # We can check if generator.reader was called batch_size times.
        # For more specific checks on *which* files, we'd need to mock np.random.choice
        self.assertEqual(generator.reader.call_count, batch_size)


    @patch('src.generators.base_generators.ThreadPool')
    def test_iteration_and_batch_processing_reader_failure(self, mock_thread_pool_class: MagicMock) -> None:
        filenames: typing.List[str] = ["file1.img", "file2.img"]
        batch_size: int = 2
        shape: typing.Tuple[int, int, int] = (64, 64, 3)

        mock_pool_instance = MagicMock()
        mock_thread_pool_class.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.side_effect = IOError("Failed to read file")

        generator = BatchGenerator(filenames=filenames, batch_size=batch_size, shape=shape)
        
        # The reader itself is called within the ThreadPool's map function.
        # So, if map raises an error (simulating a failure in one of the reader calls),
        # that error should propagate.
        generator.reader = MagicMock(side_effect=IOError("This specific mock shouldn't be directly called if map is properly mocked"))

        with self.assertRaises(IOError):
            next(generator)
            
        mock_thread_pool_class.assert_called_once_with(processes=batch_size)
        mock_pool_instance.map.assert_called_once()
        # generator.reader might not be called if the error occurs early in batch_processing
        # or if np.random.choice selects files that cause issues.
        # The key is that the error from the pool.map (simulating reader failure) propagates.

if __name__ == '__main__':
    unittest.main()
