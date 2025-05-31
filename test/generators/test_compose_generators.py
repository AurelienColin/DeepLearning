import pytest # Added for consistency, though not strictly needed by current test content
from unittest.mock import MagicMock, call
import typing
import numpy as np

from src.generators.base_generators import BatchGenerator, PostProcessGenerator, compose_generators

# Define Mock PostProcessor classes for testing composition
class MockPostProcessor1(PostProcessGenerator):
    def __init__(self, generator: typing.Union[BatchGenerator, PostProcessGenerator]):
        super().__init__(generator)

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Simple pass-through or minimal modification for testing chain
        return next(self.generator) 

class MockPostProcessor2(PostProcessGenerator):
    def __init__(self, generator: typing.Union[BatchGenerator, PostProcessGenerator]):
        super().__init__(generator)

    def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Simple pass-through or minimal modification for testing chain
        inputs, outputs = next(self.generator)
        return inputs * 2, outputs * 2 # Example modification

# Removed TestComposeGenerators class wrapper, tests are now functions

def test_no_composition() -> None:
    mock_batch_gen: BatchGenerator = MagicMock(spec=BatchGenerator)
    composed_gen = compose_generators(generator=mock_batch_gen, composition=[])
    assert composed_gen is mock_batch_gen

def test_single_composition() -> None:
    mock_batch_gen: BatchGenerator = MagicMock(spec=BatchGenerator)

    # Using a real mock class type for assertion
    composed_gen = compose_generators(generator=mock_batch_gen, composition=[MockPostProcessor1])

    assert isinstance(composed_gen, MockPostProcessor1)
    assert composed_gen.generator is mock_batch_gen

def test_multiple_compositions() -> None:
    mock_batch_gen: BatchGenerator = MagicMock(spec=BatchGenerator)

    # Using real mock class types for assertion
    composed_gen = compose_generators(generator=mock_batch_gen, composition=[MockPostProcessor1, MockPostProcessor2])

    assert isinstance(composed_gen, MockPostProcessor2)
    assert isinstance(composed_gen.generator, MockPostProcessor1)
    assert composed_gen.generator.generator is mock_batch_gen

def test_composition_instantiation_order() -> None:
    mock_batch_gen: BatchGenerator = MagicMock(spec=BatchGenerator)

    # More direct way to test instantiation if needed, using Mocks for the classes themselves
    MockPP1Class: MagicMock = MagicMock(spec=PostProcessGenerator)
    MockPP2Class: MagicMock = MagicMock(spec=PostProcessGenerator)

    # Make them return mock instances when called
    mock_pp1_instance = MagicMock(spec=PostProcessGenerator)
    MockPP1Class.return_value = mock_pp1_instance

    mock_pp2_instance = MagicMock(spec=PostProcessGenerator)
    MockPP2Class.return_value = mock_pp2_instance

    composition_list: typing.List[typing.Type[PostProcessGenerator]] = [MockPP1Class, MockPP2Class]

    composed_gen = compose_generators(generator=mock_batch_gen, composition=composition_list)

    MockPP1Class.assert_called_once_with(mock_batch_gen)
    MockPP2Class.assert_called_once_with(mock_pp1_instance)
    assert composed_gen is mock_pp2_instance
