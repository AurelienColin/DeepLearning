import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

# Mock Rignak (used by parent Plotter)
mock_rignak_display_dep = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_dep
mock_rignak_lazy_property_dep = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_dep

# Mock ModelWrapper (parent of DiffusionModelWrapper)
mock_model_wrapper_module_dep = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_dep

# Mock DiffusionModelWrapper specifically
mock_diffusion_model_wrapper_module_dep = MagicMock()
sys.modules['src.models.image_to_image.diffusion_model_wrapper'] = mock_diffusion_model_wrapper_module_dep

# Mock numpy
mock_numpy_dep = MagicMock()
sys.modules['numpy'] = mock_numpy_dep

# Import after mocks
from callbacks.plotters.image_to_image.diffusion_example_plotter import DiffusionExamplePlotter
# PlotterFromArrays is parent, Plotter is grandparent

class TestDiffusionExamplePlotter(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_dep.Display.reset_mock()
        mock_model_wrapper_module_dep.ModelWrapper.reset_mock()
        mock_diffusion_model_wrapper_module_dep.DiffusionModelWrapper.reset_mock()
        mock_numpy_dep.reset_mock()

        self.mock_diffusion_mw = MagicMock(spec=mock_diffusion_model_wrapper_module_dep.DiffusionModelWrapper)
        self.mock_keras_model = MagicMock(name="KerasModelInstance") # For self.mock_diffusion_mw.model
        self.mock_diffusion_mw.model = self.mock_keras_model
        self.mock_diffusion_mw.noise_factor = 0.5 # Example value

        # Inputs and outputs for PlotterFromArrays
        self.num_samples_original = 5 # More than ncols=4 to test slicing
        self.input_height, self.input_width, self.input_channels = 32, 32, 3
        
        self.mock_inputs_original = MagicMock(name="inputs_original_array")
        self.mock_inputs_original.shape = [self.num_samples_original, self.input_height, self.input_width, self.input_channels]
        
        self.mock_outputs_original = MagicMock(name="outputs_original_array")
        self.mock_outputs_original.shape = [self.num_samples_original, self.input_height, self.input_width, self.input_channels]

        # Mock slicing behavior for inputs/outputs in __init__
        self.mock_inputs_sliced = MagicMock(name="inputs_sliced")
        self.mock_inputs_sliced.shape = [min(self.num_samples_original, 4), self.input_height, self.input_width, self.input_channels]
        self.mock_inputs_original.__getitem__.return_value = self.mock_inputs_sliced
        
        self.mock_outputs_sliced = MagicMock(name="outputs_sliced")
        self.mock_outputs_original.__getitem__.return_value = self.mock_outputs_sliced
        
        # Mock numpy.arange, np.reshape, np.repeat for diffusion_times
        self.mock_arange_val = MagicMock(name="arange_val")
        mock_numpy_dep.arange.return_value = self.mock_arange_val
        self.mock_reshape_val = MagicMock(name="reshape_val")
        # Make reshape_val iterable for the loop in __call__
        self.mock_reshape_val_iter = [MagicMock(name=f"dt_row_{i}") for i in range(8)] # nrows=8
        self.mock_reshape_val.__iter__.return_value = iter(self.mock_reshape_val_iter)
        self.mock_reshape_val.shape = [8,1,1,1,1] # nrows, then 1,1,1,1 for broadcasting
        mock_numpy_dep.reshape.return_value = self.mock_reshape_val
        self.mock_repeat_val = MagicMock(name="repeat_val") # This is what's iterated in __call__
        self.mock_repeat_val.__iter__.return_value = iter(self.mock_reshape_val_iter) # Use the same iterable
        mock_numpy_dep.repeat.return_value = self.mock_repeat_val


    def create_plotter(self):
        return DiffusionExamplePlotter(
            inputs=self.mock_inputs_original, # Original, gets sliced in __init__
            outputs=self.mock_outputs_original, # Original, gets sliced in __init__
            model_wrapper=self.mock_diffusion_mw
        )

    def test_initialization(self):
        plotter = self.create_plotter()

        expected_ncols = min(self.num_samples_original, 4)
        expected_nrows = 8
        
        # Check slicing in __init__
        self.mock_inputs_original.__getitem__.assert_called_once_with(slice(None, expected_ncols))
        self.mock_outputs_original.__getitem__.assert_called_once_with(slice(None, 4)) # Hardcoded 4 for outputs slice

        # Attributes from PlotterFromArrays (should be the sliced versions)
        self.assertIs(plotter.inputs, self.mock_inputs_sliced)
        self.assertIs(plotter.outputs, self.mock_outputs_sliced) # Corrected: outputs also get sliced
        
        # Attributes from Plotter (grandparent)
        self.assertIs(plotter.model_wrapper, self.mock_diffusion_mw)
        self.assertEqual(plotter.ncols, expected_ncols)
        self.assertEqual(plotter.nrows, expected_nrows)
        self.assertEqual(plotter.thumbnail_size, (16, 4)) # (4*4, 4)

    @patch.object(DiffusionExamplePlotter, 'concatenate', return_value=MagicMock(name="ConcatenatedImagesRow"))
    @patch.object(DiffusionExamplePlotter, 'imshow') # Mock from Plotter
    @patch.object(DiffusionExamplePlotter, 'display', new_callable=MagicMock) # Mock LazyProperty
    def test_call_method_loop_and_plotting(self, mock_display_prop, mock_imshow_method, mock_concatenate_method):
        plotter = self.create_plotter() # This sets up plotter.inputs to be self.mock_inputs_sliced

        # Mock np.random.normal
        mock_initial_noises_val = MagicMock(name="InitialNoises")
        mock_numpy_dep.random.normal.return_value = mock_initial_noises_val

        # Mock diffusion_schedule from model_wrapper
        mock_noise_rates_tensor = MagicMock(name="NoiseRatesTensor")
        mock_signal_rates_tensor = MagicMock(name="SignalRatesTensor")
        self.mock_diffusion_mw.diffusion_schedule.return_value = (mock_noise_rates_tensor, mock_signal_rates_tensor)

        # Mock noisy_images calculation (signal_rates * self.inputs + noise_rates * noises)
        # This involves tensor math. For simplicity, assume it produces a mock.
        mock_noisy_images_calc = MagicMock(name="NoisyImagesCalculated")
        # To mock `A * B + C * D`, we'd need to mock __mul__ and __add__ on these tensors.
        # Let's assume signal_rates * self.inputs results in one mock, and noise_rates * noises in another,
        # and their sum is mock_noisy_images_calc.
        # For now, we'll check that the components are used.

        # Mock model prediction
        mock_pred_images_raw = MagicMock(name="PredImagesRawTensor")
        mock_pred_images_numpy = MagicMock(name="PredImagesNumpy")
        mock_pred_images_raw.numpy.return_value = mock_pred_images_numpy # model(...).numpy()
        self.mock_keras_model.return_value = mock_pred_images_raw
        # Slicing after .numpy() -> pred_images_numpy[:,:,:,:3]
        mock_pred_images_sliced_for_error = MagicMock(name="PredImagesSlicedForError")
        mock_pred_images_numpy.__getitem__.return_value = mock_pred_images_sliced_for_error


        # Mock error calculation: np.abs(pred_images_sliced_for_error - self.inputs)
        mock_error_val = MagicMock(name="ErrorValue")
        mock_numpy_dep.abs.return_value = mock_error_val

        # Mock concatenate return for each row
        mock_concatenated_row_list = [MagicMock(name=f"ConcatImg_Col{j}") for j in range(plotter.ncols)]
        mock_concatenate_method.return_value = mock_concatenated_row_list
        
        # Call the method
        returned_display = plotter()

        # 1. np.random.normal call
        mock_numpy_dep.random.normal.assert_called_once_with(size=self.mock_inputs_sliced.shape) # Uses sliced inputs shape
        
        # 2. diffusion_times setup
        mock_numpy_dep.arange.assert_called_once_with(0, 1.001, 1 / (plotter.nrows - 1))
        mock_numpy_dep.reshape.assert_called_once_with(self.mock_arange_val, (plotter.nrows, 1, 1, 1, 1))
        mock_numpy_dep.repeat.assert_called_once_with(self.mock_reshape_val, self.mock_inputs_sliced.shape[0], axis=1)

        # Loop assertions (nrows times)
        self.assertEqual(self.mock_diffusion_mw.diffusion_schedule.call_count, plotter.nrows)
        self.assertEqual(self.mock_keras_model.call_count, plotter.nrows)
        self.assertEqual(mock_numpy_dep.abs.call_count, plotter.nrows) # For error calc
        self.assertEqual(mock_concatenate_method.call_count, plotter.nrows)
        self.assertEqual(mock_imshow_method.call_count, plotter.nrows * plotter.ncols)

        # Example check for one iteration (e.g., first iteration, i_row=0)
        first_diffusion_time_row = self.mock_reshape_val_iter[0] # from mock_repeat_val's iterator source
        self.mock_diffusion_mw.diffusion_schedule.assert_any_call(first_diffusion_time_row)
        
        # Check arguments to model call for one iteration (assuming noisy_images was formed correctly)
        # self.mock_keras_model.assert_any_call([ANY_noisy_images, mock_noise_rates_tensor], training=False)
        # This needs a more specific mock for noisy_images if we want to assert it.

        # Check inputs to concatenate for one iteration
        # For the first iteration, noisy_images would be:
        # mock_signal_rates_tensor * plotter.inputs + mock_noise_rates_tensor * mock_initial_noises_val
        # We'd need to mock the tensor arithmetic to get a specific mock for noisy_images.
        # For now, check that plotter.inputs and mock_initial_noises_val are involved.
        mock_concatenate_method.assert_any_call(
            plotter.inputs, # This is self.mock_inputs_sliced
            ANY, # noisy_images
            mock_pred_images_sliced_for_error, # Sliced predictions
            mock_error_val
        )
        
        # Check imshow calls for one row
        for j_col in range(plotter.ncols):
            mock_imshow_method.assert_any_call(0 * plotter.ncols + j_col, mock_concatenated_row_list[j_col])
            
        self.assertIs(returned_display, mock_display_prop)


if __name__ == '__main__':
    unittest.main()
