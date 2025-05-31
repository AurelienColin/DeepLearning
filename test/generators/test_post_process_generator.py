import pytest # Added for pytest.raises
from unittest.mock import MagicMock
import typing
import numpy as np

from src.generators.base_generators import PostProcessGenerator, BatchGenerator
from src.output_spaces.output_space import OutputSpace

# Removed TestPostProcessGenerator class wrapper, tests are now functions

def test_instantiation() -> None:
    mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
    post_processor = PostProcessGenerator(generator=mock_batch_generator)
    assert isinstance(post_processor, PostProcessGenerator)
    assert post_processor.generator == mock_batch_generator

def test_output_space_property() -> None:
    mock_output_space: OutputSpace = MagicMock(spec=OutputSpace)
    mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
    mock_batch_generator.output_space = mock_output_space

    post_processor = PostProcessGenerator(generator=mock_batch_generator)
    assert post_processor.output_space == mock_output_space

def test_batch_size_property() -> None:
    expected_batch_size: int = 4
    mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
    mock_batch_generator.batch_size = expected_batch_size

    post_processor = PostProcessGenerator(generator=mock_batch_generator)
    assert post_processor.batch_size == expected_batch_size

def test_iteration_raises_not_implemented() -> None:
    mock_batch_generator: BatchGenerator = MagicMock(spec=BatchGenerator)
    # Ensure the mock generator itself is iterable and its __next__ can be called by PostProcessGenerator's __next__
    # although the base PostProcessGenerator.__next__ should raise NotImplementedError before that.
    mock_batch_generator.__iter__.return_value = mock_batch_generator
    mock_batch_generator.__next__.return_value = (np.array([]), np.array([])) # Dummy return

    post_processor = PostProcessGenerator(generator=mock_batch_generator)

    # Make post_processor iterable
    post_processor = iter(post_processor)

    with pytest.raises(NotImplementedError):
        next(post_processor)
