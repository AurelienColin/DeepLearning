import tempfile
import numpy as np # Retained for TestAutoEncodingSample & np.testing

# Import AutoEncodingSample - Specific to this file now
from src.samples.image_to_image.autoencoder_sample import AutoEncodingSample
# Import create_dummy_image from the new main test file
from test.test_sample_main import create_dummy_image


def test_output_data() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy RGB image using the imported function
        dummy_image_path = create_dummy_image(
            tmpdir,
            filename="test_ae_image.png",
            size=(32, 32), # Original image size
            mode='RGB',
            color='purple'
        )

        # Instantiate AutoEncodingSample, shape matches original for this base test
        sample = AutoEncodingSample(input_filename=dummy_image_path, shape=(32, 32, 3))

        # Assert that sample.output_data is a np.ndarray
        assert isinstance(sample.output_data, np.ndarray)

        # Assert that sample.output_data.shape is the same as sample.input_data.shape
        assert sample.output_data.shape == sample.input_data.shape
        assert sample.output_data.shape == (32, 32, 3)


        # Assert that sample.output_data.dtype is np.float32
        assert sample.output_data.dtype == np.float32

        # Assert that the arrays are identical
        np.testing.assert_array_equal(sample.output_data, sample.input_data)

        # Additional check for value range
        assert np.all((sample.output_data >= 0) & (sample.output_data <= 1))
        assert np.all((sample.input_data >= 0) & (sample.input_data <= 1))

def test_output_data_grayscale() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_gray_path = create_dummy_image(
            tmpdir,
            filename="test_ae_gray.png",
            size=(32, 32), # Original image size
            mode='L',      # Grayscale mode
            color='gray'
        )

        target_shape = (32, 32, 1)
        sample = AutoEncodingSample(input_filename=dummy_gray_path, shape=target_shape)

        # Access data
        input_data = sample.input_data
        output_data = sample.output_data

        # Assertions for output_data
        assert isinstance(output_data, np.ndarray)
        assert output_data.shape == target_shape
        assert output_data.dtype == np.float32
        assert np.all((output_data >= 0) & (output_data <= 1))

        # Assertions for input_data (to ensure it's also correctly processed)
        assert isinstance(input_data, np.ndarray)
        assert input_data.shape == target_shape
        assert input_data.dtype == np.float32
        assert np.all((input_data >= 0) & (input_data <= 1))

        # Assert that output_data is identical to input_data
        np.testing.assert_array_equal(output_data, input_data)

def test_output_data_different_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a standard RGB dummy image (e.g., 64x64)
        dummy_rgb_path = create_dummy_image(
            tmpdir,
            filename="test_ae_rgb_shape.png",
            size=(64, 64),
            mode='RGB',
            color='cyan'
        )

        # Test Case 1: Non-square shape (32, 48, 3)
        shape1 = (32, 48, 3) # Height, Width, Channels
        sample1 = AutoEncodingSample(input_filename=dummy_rgb_path, shape=shape1)

        input_data1 = sample1.input_data
        output_data1 = sample1.output_data

        assert input_data1.shape == shape1, "Input data shape mismatch for shape1"
        assert isinstance(output_data1, np.ndarray)
        assert output_data1.shape == shape1, "Output data shape mismatch for shape1"
        assert output_data1.dtype == np.float32
        np.testing.assert_array_equal(output_data1, input_data1, "Output not equal to input for shape1")
        assert np.all((output_data1 >= 0) & (output_data1 <= 1))

        # Test Case 2: Another non-square shape (48, 32, 3)
        shape2 = (48, 32, 3) # Height, Width, Channels
        sample2 = AutoEncodingSample(input_filename=dummy_rgb_path, shape=shape2)

        input_data2 = sample2.input_data
        output_data2 = sample2.output_data

        assert input_data2.shape == shape2, "Input data shape mismatch for shape2"
        assert isinstance(output_data2, np.ndarray)
        assert output_data2.shape == shape2, "Output data shape mismatch for shape2"
        assert output_data2.dtype == np.float32
        np.testing.assert_array_equal(output_data2, input_data2, "Output not equal to input for shape2")
        assert np.all((output_data2 >= 0) & (output_data2 <= 1))
