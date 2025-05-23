import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import os

# Adjust the path to import generators from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generators.base_generators import BatchGenerator, PostProcessGenerator, compose_generators

class TestBatchGenerator(unittest.TestCase):

    def test_batch_generator_initialization(self):
        filenames = ['file1.jpg', 'file2.jpg', 'file3.jpg']
        batch_size = 2
        output_shape = (128, 128, 1)
        input_shape = (128, 128, 3)

        gen = BatchGenerator(
            filenames=filenames,
            batch_size=batch_size,
            output_shape=output_shape,
            input_shape=input_shape,
            shuffle=True,
            name="test_gen"
        )

        self.assertEqual(gen.filenames, filenames)
        self.assertEqual(gen.batch_size, batch_size)
        self.assertEqual(gen.output_shape, output_shape)
        self.assertEqual(gen.input_shape, input_shape)
        self.assertTrue(gen.shuffle)
        self.assertEqual(gen.name, "test_gen")
        self.assertEqual(len(gen), (len(filenames) + batch_size - 1) // batch_size) # ceiling division

    def test_reader_not_implemented(self):
        gen = BatchGenerator(['file1'], 1, (10,), (10,))
        with self.assertRaises(NotImplementedError):
            gen.reader('file1')

    def test_batch_processing_calls_reader(self):
        filenames = ['file1.png', 'file2.png']
        batch_size = 2
        output_shape = (64, 1)
        input_shape = (64, 3)

        gen = BatchGenerator(filenames, batch_size, output_shape, input_shape)
        
        # Mock the reader method
        mock_input_data = np.random.rand(input_shape[0], input_shape[1])
        mock_output_data = np.random.rand(output_shape[0], output_shape[1])
        gen.reader = MagicMock(return_value=(mock_input_data, mock_output_data))

        batch_x, batch_y = gen.batch_processing(filenames)

        gen.reader.assert_has_calls([call('file1.png'), call('file2.png')])
        self.assertEqual(batch_x.shape, (batch_size, input_shape[0], input_shape[1]))
        self.assertEqual(batch_y.shape, (batch_size, output_shape[0], output_shape[1]))
        np.testing.assert_array_equal(batch_x[0], mock_input_data)
        np.testing.assert_array_equal(batch_y[1], mock_output_data)


    def test_next_batch_generation(self):
        filenames = ['f1', 'f2', 'f3', 'f4', 'f5']
        batch_size = 2
        output_shape = (1,)
        input_shape = (1,)
        gen = BatchGenerator(filenames, batch_size, output_shape, input_shape, shuffle=False)

        # Mock batch_processing to avoid reader dependency
        def mock_bp(batch_files):
            # Return arrays of correct batch size, content doesn't matter for this test
            return np.zeros((len(batch_files), *input_shape)), \
                   np.zeros((len(batch_files), *output_shape))
        gen.batch_processing = MagicMock(side_effect=mock_bp)

        # Epoch 1
        batch1_x, batch1_y = next(gen)
        self.assertEqual(batch1_x.shape, (batch_size, *input_shape))
        self.assertEqual(batch1_y.shape, (batch_size, *output_shape))
        gen.batch_processing.assert_called_with(['f1', 'f2'])

        batch2_x, batch2_y = next(gen)
        self.assertEqual(batch2_x.shape, (batch_size, *input_shape))
        gen.batch_processing.assert_called_with(['f3', 'f4'])
        
        batch3_x, batch3_y = next(gen) # Last batch, might be smaller
        self.assertEqual(batch3_x.shape, (1, *input_shape)) # Remaining file
        gen.batch_processing.assert_called_with(['f5'])

        # Epoch 2 - should reshuffle if shuffle=True, or restart if shuffle=False
        gen.on_epoch_end() # Manually call for testing, Keras does this
        batch1_e2_x, _ = next(gen)
        self.assertEqual(batch1_e2_x.shape, (batch_size, *input_shape))
        gen.batch_processing.assert_called_with(['f1', 'f2']) # shuffle is False

    def test_shuffle_on_epoch_end(self):
        filenames = np.array(['f1', 'f2', 'f3', 'f4'])
        gen = BatchGenerator(filenames.tolist(), 2, (1,), (1,), shuffle=True)
        
        # Keep track of original filenames order for one iter
        original_order_batch1 = gen.filenames[:2]
        original_order_batch2 = gen.filenames[2:4]

        # Mock batch_processing
        gen.batch_processing = MagicMock(return_value=(np.zeros((2,1)), np.zeros((2,1))))
        
        next(gen)
        gen.batch_processing.assert_called_with(original_order_batch1)
        next(gen)
        gen.batch_processing.assert_called_with(original_order_batch2)

        # Now call on_epoch_end, filenames should be shuffled
        with patch('numpy.random.shuffle') as mock_shuffle:
            gen.on_epoch_end()
            mock_shuffle.assert_called_once()
            # Note: We can't easily check the content of gen.filenames after shuffle
            # because the shuffle is in-place and random.
            # The mock_shuffle assertion is the main check here.

class TestPostProcessGenerator(unittest.TestCase):

    def test_post_process_generator_initialization(self):
        mock_inner_generator = MagicMock(spec=BatchGenerator)
        mock_inner_generator.__len__ = MagicMock(return_value=10) # Mock __len__

        pp_gen = PostProcessGenerator(mock_inner_generator)
        self.assertEqual(pp_gen.generator, mock_inner_generator)
        self.assertEqual(len(pp_gen), 10)
        mock_inner_generator.__len__.assert_called_once()

    def test_next_not_implemented(self):
        mock_inner_generator = MagicMock(spec=BatchGenerator)
        # Simulate the inner generator returning a batch
        mock_inner_generator.__next__ = MagicMock(return_value=(np.array([1]), np.array([2])))
        
        pp_gen = PostProcessGenerator(mock_inner_generator)
        with self.assertRaises(NotImplementedError):
            next(pp_gen)
        mock_inner_generator.__next__.assert_called_once() # Ensure it tried to get from inner

    def test_on_epoch_end_calls_inner(self):
        mock_inner_generator = MagicMock(spec=BatchGenerator)
        mock_inner_generator.on_epoch_end = MagicMock()
        pp_gen = PostProcessGenerator(mock_inner_generator)
        pp_gen.on_epoch_end()
        mock_inner_generator.on_epoch_end.assert_called_once()


class TestComposeGenerators(unittest.TestCase):

    def test_compose_generators_chaining(self):
        # Create mock generators
        gen1_output_x = np.array([[1,2]])
        gen1_output_y = np.array([[3,4]])
        mock_gen1 = MagicMock()
        mock_gen1.__next__ = MagicMock(return_value=(gen1_output_x, gen1_output_y))
        mock_gen1.__len__ = MagicMock(return_value=5)

        gen2_output_x = np.array([[5,6]])
        gen2_output_y = np.array([[7,8]])
        mock_gen2 = MagicMock()
        # gen2 will take output of gen1 as its input conceptually
        # For this test, we just ensure it's called and its output is the final one
        mock_gen2.process_batch = MagicMock(return_value=(gen2_output_x, gen2_output_y))
        
        # Wrap gen2 in a PostProcessGenerator structure for compose_generators
        # This requires gen2 to be an instance of PostProcessGenerator or have a generator attribute
        # For simplicity, let's make a dummy PostProcessGenerator that uses gen2.process_batch
        class SimplePPG(PostProcessGenerator):
            def __init__(self, generator, processor_mock):
                super().__init__(generator)
                self.processor_mock = processor_mock
            def __next__(self):
                x, y = next(self.generator)
                return self.processor_mock(x, y)

        pp_gen2 = SimplePPG(mock_gen1, mock_gen2.process_batch)
        
        composed_gen = compose_generators(mock_gen1, pp_gen2)

        # Test __len__
        self.assertEqual(len(composed_gen), 5) # Should be len of the base generator

        # Test __next__
        final_x, final_y = next(composed_gen)
        
        mock_gen1.__next__.assert_called_once()
        # pp_gen2's __next__ calls its generator's next (mock_gen1 again if not careful)
        # The way compose_generators re-assigns, it should be:
        # composed_gen.generator is mock_gen1
        # composed_gen.post_process_generator is pp_gen2
        # pp_gen2.generator is ALSO mock_gen1 (this is how PostProcessGen works)

        # In compose_generators, the `post_process_generator.generator` is reassigned to `base_generator`.
        # So, when `next(composed_gen)` is called, it calls `next(pp_gen2)`,
        # which in turn calls `next(mock_gen1)` then `mock_gen2.process_batch`.
        
        mock_gen2.process_batch.assert_called_once_with(gen1_output_x, gen1_output_y)
        np.testing.assert_array_equal(final_x, gen2_output_x)
        np.testing.assert_array_equal(final_y, gen2_output_y)

    def test_compose_generators_on_epoch_end(self):
        mock_gen1 = MagicMock()
        mock_gen1.on_epoch_end = MagicMock()
        
        mock_pp_gen = MagicMock(spec=PostProcessGenerator)
        mock_pp_gen.on_epoch_end = MagicMock() # This will be called by ComposedGenerator

        composed = compose_generators(mock_gen1, mock_pp_gen)
        composed.on_epoch_end()

        # The ComposedGenerator's on_epoch_end calls the post_process_generator's on_epoch_end.
        # The PostProcessGenerator's on_epoch_end, in turn, calls its internal generator's on_epoch_end.
        # In the composed setup, pp_gen.generator becomes gen1.
        mock_pp_gen.on_epoch_end.assert_called_once()
        # Check if the PostProcessGenerator's on_epoch_end correctly calls the base generator's on_epoch_end
        # This requires pp_gen.generator.on_epoch_end() to be called inside mock_pp_gen.on_epoch_end
        # For this, we assume PostProcessGenerator's implementation is correct.
        # If we were testing PostProcessGenerator itself more deeply, we'd mock its internal call.


if __name__ == '__main__':
    unittest.main()
