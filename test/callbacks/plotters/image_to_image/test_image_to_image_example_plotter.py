import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

# Mock Rignak (used by parent Plotter)
mock_rignak_display_i2i_ep = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_i2i_ep
mock_rignak_lazy_property_i2i_ep = MagicMock() # For @LazyProperty
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_i2i_ep


# Mock ModelWrapper (passed to Plotter's __init__)
mock_model_wrapper_module_i2i_ep = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_i2i_ep

# Mock numpy (used by Plotter.concatenate and in ImageToImageExamplePlotter's __init__)
mock_numpy_i2i_ep = MagicMock()
sys.modules['numpy'] = mock_numpy_i2i_ep

# Import after mocks
from callbacks.plotters.image_to_image.image_to_image_example_plotter import ImageToImageExamplePlotter
# PlotterFromArrays is the parent, Plotter is the grandparent.

class TestImageToImageExamplePlotter(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_i2i_ep.Display.reset_mock()
        mock_model_wrapper_module_i2i_ep.ModelWrapper.reset_mock()
        mock_numpy_i2i_ep.reset_mock()

        self.mock_mw_instance = MagicMock(spec=mock_model_wrapper_module_i2i_ep.ModelWrapper)
        # Mock the Keras model that model_wrapper.model would return
        self.mock_keras_model = MagicMock(name="KerasModelInstance")
        self.mock_mw_instance.model = self.mock_keras_model
        
        # Inputs and outputs for PlotterFromArrays
        self.num_samples = 7 # Example, results in 2 rows if ncols=4
        self.input_height, self.input_width, self.input_channels = 32, 32, 3
        
        # Mock np.ndarray type for spec if needed, or just use MagicMock
        self.mock_inputs_arr = MagicMock(name="inputs_array")
        self.mock_inputs_arr.shape = [self.num_samples, self.input_height, self.input_width, self.input_channels]
        
        self.mock_outputs_arr = MagicMock(name="outputs_array")
        # Output shape should match input for this plotter's error calculation logic
        self.mock_outputs_arr.shape = [self.num_samples, self.input_height, self.input_width, self.input_channels]

        # Mock numpy.ceil for nrows calculation in __init__
        mock_numpy_i2i_ep.ceil.return_value = float( (self.num_samples + 4 - 1) // 4 ) # Manual ceil for testing

    def create_plotter(self):
        return ImageToImageExamplePlotter(
            inputs=self.mock_inputs_arr,
            outputs=self.mock_outputs_arr,
            model_wrapper=self.mock_mw_instance
        )

    def test_initialization(self):
        plotter = self.create_plotter()

        expected_ncols = 4
        expected_nrows = int(mock_numpy_i2i_ep.ceil.return_value) # 2 for 7 samples
        
        # From PlotterFromArrays
        self.assertIs(plotter.inputs, self.mock_inputs_arr)
        self.assertIs(plotter.outputs, self.mock_outputs_arr)
        
        # From Plotter (grandparent) via ImageToImageExamplePlotter's super() call
        self.assertIs(plotter.model_wrapper, self.mock_mw_instance)
        self.assertEqual(plotter.ncols, expected_ncols)
        self.assertEqual(plotter.nrows, expected_nrows)
        
        # Check thumbnail_size modification
        # Default from Plotter is (4,4). I2IEP does: (super().thumbnail_size[0] * 4, super().thumbnail_size[1])
        # So, (4*4, 4) = (16,4)
        self.assertEqual(plotter.thumbnail_size, (16, 4))
        
        mock_numpy_i2i_ep.ceil.assert_called_once_with(self.num_samples / expected_ncols)


    @patch.object(ImageToImageExamplePlotter, 'concatenate', return_value=MagicMock(name="ConcatenatedImages"))
    @patch.object(ImageToImageExamplePlotter, 'imshow') # Mock from Plotter
    @patch.object(ImageToImageExamplePlotter, 'display', new_callable=MagicMock) # Mock LazyProperty
    def test_call_method(self, mock_display_prop, mock_imshow_method, mock_concatenate_method):
        plotter = self.create_plotter()
        
        # Mock predictions from the model
        mock_pred_images_raw = MagicMock(name="PredImagesRawTensor")
        mock_pred_images_numpy = MagicMock(name="PredImagesNumpy")
        mock_pred_images_raw.numpy.return_value = mock_pred_images_numpy
        self.mock_keras_model.return_value = mock_pred_images_raw # model(inputs)

        # Mock error calculation: error = np.abs(pred_images[:,:,:,:3] - self.outputs[:,:,:,:3])
        # This involves slicing and subtraction on numpy arrays (or mocks behaving like them)
        mock_pred_sliced = MagicMock(name="PredSlicedForError")
        mock_outputs_sliced = MagicMock(name="OutputsSlicedForError")
        mock_pred_images_numpy.__getitem__.return_value = mock_pred_sliced
        self.mock_outputs_arr.__getitem__.return_value = mock_outputs_sliced
        
        mock_error_calc_result = MagicMock(name="ErrorCalculationResult")
        mock_numpy_i2i_ep.abs.return_value = mock_error_calc_result # np.abs(...)

        # Mock concatenate return value (already done by @patch)
        mock_concatenated_images_list = [MagicMock(name="Img1"), MagicMock(name="Img2")] # List of images
        mock_concatenate_method.return_value = mock_concatenated_images_list
        
        # Call the method
        returned_display = plotter()

        # 1. Check model prediction call
        self.mock_keras_model.assert_called_once_with(self.mock_inputs_arr, training=False)
        mock_pred_images_raw.numpy.assert_called_once()

        # 2. Check error calculation slicing
        slicer_arg = (slice(None), slice(None), slice(None), slice(None, 3))
        mock_pred_images_numpy.__getitem__.assert_called_once_with(slicer_arg)
        self.mock_outputs_arr.__getitem__.assert_called_once_with(slicer_arg)
        mock_numpy_i2i_ep.abs.assert_called_once_with(mock_pred_sliced - mock_outputs_sliced)

        # 3. Check self.concatenate call
        mock_concatenate_method.assert_called_once_with(
            self.mock_inputs_arr, self.mock_outputs_arr, mock_pred_images_numpy, mock_error_calc_result
        )

        # 4. Check self.imshow calls for each image from concatenate
        expected_imshow_calls = []
        for i, img_mock in enumerate(mock_concatenated_images_list):
            expected_imshow_calls.append(call(i, img_mock))
        mock_imshow_method.assert_has_calls(expected_imshow_calls)
        self.assertEqual(mock_imshow_method.call_count, len(mock_concatenated_images_list))

        # 5. Check if display is returned
        self.assertIs(returned_display, mock_display_prop)
        
        # 6. Check reset_display decorator effect (_display should be None before __call__)
        # The decorator sets _display to None *before* the method runs.
        # If display property was accessed, it would be re-initialized.
        # This plotter's __call__ returns self.display, so the display property is accessed.
        self.assertIsNotNone(plotter._display) # It's accessed and cached by the end of __call__


if __name__ == '__main__':
    unittest.main()
