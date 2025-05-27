import unittest
from unittest.mock import MagicMock
import typing
import numpy as np

from src.generators.base_generators import PostProcessGenerator, BatchGenerator
from src.output_spaces.output_space import OutputSpace

class TestPostProcessGenerator(unittest.TestCase):

    def test_instantiation(self) -> None:
        mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
        post_processor = PostProcessGenerator(generator=mock_batch_generator)
        self.assertIsInstance(post_processor, PostProcessGenerator)
        self.assertEqual(post_processor.generator, mock_batch_generator)

    def test_output_space_property(self) -> None:
        mock_output_space: OutputSpace = MagicMock(spec=OutputSpace)
        mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
        mock_batch_generator.output_space = mock_output_space
        
        post_processor = PostProcessGenerator(generator=mock_batch_generator)
        self.assertEqual(post_processor.output_space, mock_output_space)

    def test_batch_size_property(self) -> None:
        expected_batch_size: int = 4
        mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
        mock_batch_generator.batch_size = expected_batch_size
        
        post_processor = PostProcessGenerator(generator=mock_batch_generator)
        self.assertEqual(post_processor.batch_size, expected_batch_size)

    def test_iteration_raises_not_implemented(self) -> None:
        mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
        # Ensure the mock generator itself is iterable and its __next__ can be called by PostProcessGenerator's __next__
        # although the base PostProcessGenerator.__next__ should raise NotImplementedError before that.
        mock_batch_generator.__iter__.return_value = mock_batch_generator
        mock_batch_generator.__next__.return_value = (np.array([]), np.array([])) # Dummy return

        post_processor = PostProcessGenerator(generator=mock_batch_generator)
        
        # Make post_processor iterable
        post_processor = iter(post_processor)

        with self.assertRaises(NotImplementedError):
            next(post_processor)

if __name__ == '__main__':
    unittest.main()
