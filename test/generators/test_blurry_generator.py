import pytest # Added for pytest.approx
from unittest.mock import MagicMock, patch
import numpy as np
import typing

from src.generators.blurry import BlurryGenerator
from src.generators.base_generators import PostProcessGenerator, BatchGenerator # Or any base generator BlurryGenerator might wrap

# Removed TestBlurryGenerator class wrapper, tests are now functions

def test_instantiation() -> None:
    mock_inner_generator: MagicMock = MagicMock(spec=PostProcessGenerator) # Can be BatchGenerator or another PostProcessGenerator
    blurry_gen = BlurryGenerator(generator=mock_inner_generator)
    assert isinstance(blurry_gen, BlurryGenerator)
    assert blurry_gen.generator == mock_inner_generator

@patch('src.generators.blurry.ndimage.gaussian_filter')
def test_blur_method(mock_gaussian_filter: MagicMock) -> None:
    blurry_gen = BlurryGenerator(generator=MagicMock(spec=PostProcessGenerator)) # Dummy generator

    height, width, channels = 28, 28, 1
    input_array: np.ndarray = np.random.rand(height, width, channels)
    sigma: float = 1.5

    # Make gaussian_filter return a modified array to check it was called
    mock_gaussian_filter.return_value = input_array * 0.5

    blurred_array = blurry_gen.blur(input_array, sigma)

    assert isinstance(blurred_array, np.ndarray)
    assert blurred_array.shape == input_array.shape
    mock_gaussian_filter.assert_called_once()
    # Check sigma and axes passed to gaussian_filter
    args, kwargs = mock_gaussian_filter.call_args
    assert args[1] == pytest.approx(sigma) # args[0] is the array
    assert kwargs.get('axes') == (0,1)
    # Check if output is different (assuming filter changes it)
    assert not np.array_equal(blurred_array, input_array)


@patch.object(BlurryGenerator, 'blur')
def test_next_method(mock_blur_method: MagicMock) -> None:
    batch_size: int = 2
    height, width, channels = 32, 32, 3

    # Mock the generator that BlurryGenerator wraps
    mock_inner_generator: MagicMock = MagicMock(spec=PostProcessGenerator)

    # Define what the wrapped generator returns
    original_inputs: np.ndarray = np.random.rand(batch_size, height, width, channels).astype(np.float32)
    original_outputs: np.ndarray = np.random.rand(batch_size, height, width, channels).astype(np.float32)
    mock_inner_generator.__next__.return_value = (original_inputs.copy(), original_outputs.copy())
    # Make the mock generator iterable
    mock_inner_generator.__iter__.return_value = mock_inner_generator

    blurry_gen = BlurryGenerator(generator=mock_inner_generator)

    # Define what our mocked blur method will return
    # It should be called twice (once for inputs, once for outputs for each item in batch effectively, but simplified here)
    # For simplicity, let's assume it returns the array slightly modified
    def blur_side_effect(array: np.ndarray, sigma: float) -> np.ndarray:
        return array * (1 - sigma/ (blurry_gen.SIGMAX_MAX * 2) ) # Simulate some blurring based on sigma

    mock_blur_method.side_effect = blur_side_effect

    processed_inputs, processed_outputs = next(blurry_gen)

    assert isinstance(processed_inputs, np.ndarray)
    assert isinstance(processed_outputs, np.ndarray)

    assert processed_inputs.shape == (batch_size, height, width, channels)
    # Output has an extra channel for sigma
    assert processed_outputs.shape == (batch_size, height, width, channels + 1)

    # Check blur method was called for inputs and outputs
    # It's called per item, so batch_size * 2 times
    assert mock_blur_method.call_count == batch_size * 2

    # Check that the sigma channel in output is within [0, 1]
    sigma_channel: np.ndarray = processed_outputs[..., -1]
    assert np.all(sigma_channel >= 0)
    assert np.all(sigma_channel <= 1)
    assert sigma_channel.shape == (batch_size, height, width)
