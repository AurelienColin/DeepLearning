import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

# Mock Rignak (used by parent Plotter)
mock_rignak_display_drp = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_drp
mock_rignak_lazy_property_drp = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_drp

# Mock ModelWrapper (parent of DiffusionModelWrapper)
mock_model_wrapper_module_drp = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_drp

# Mock DiffusionModelWrapper specifically
mock_diffusion_model_wrapper_module_drp = MagicMock()
sys.modules['src.models.image_to_image.diffusion_model_wrapper'] = mock_diffusion_model_wrapper_module_drp

# Mock numpy
mock_numpy_drp = MagicMock() # Not directly used by DRP, but parent PlotterFromArrays takes np.ndarray
sys.modules['numpy'] = mock_numpy_drp


# Import after mocks
from callbacks.plotters.image_to_image.diffusion_random_plotter import DiffusionRandomPlotter
# PlotterFromArrays is parent, Plotter is grandparent

class TestDiffusionRandomPlotter(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_drp.Display.reset_mock()
        mock_model_wrapper_module_drp.ModelWrapper.reset_mock()
        mock_diffusion_model_wrapper_module_drp.DiffusionModelWrapper.reset_mock()
        mock_numpy_drp.reset_mock() # Reset numpy mock

        self.mock_diffusion_mw = MagicMock(spec=mock_diffusion_model_wrapper_module_drp.DiffusionModelWrapper)
        
        # Inputs and outputs for PlotterFromArrays init (though not directly used by DRP's call)
        self.num_input_samples = 3 # This will be ncols
        self.input_height, self.input_width, self.input_channels = 32, 32, 3
        
        self.mock_inputs_arr = MagicMock(name="inputs_array_for_drp")
        self.mock_inputs_arr.shape = [self.num_input_samples, self.input_height, self.input_width, self.input_channels]
        
        self.mock_outputs_arr = MagicMock(name="outputs_array_for_drp") # Not used by __call__

    def create_plotter(self):
        return DiffusionRandomPlotter(
            inputs=self.mock_inputs_arr,
            outputs=self.mock_outputs_arr, 
            model_wrapper=self.mock_diffusion_mw
        )

    def test_initialization(self):
        plotter = self.create_plotter()

        expected_ncols = self.num_input_samples
        expected_nrows = 5
        
        # Attributes from PlotterFromArrays
        self.assertIs(plotter.inputs, self.mock_inputs_arr)
        self.assertIs(plotter.outputs, self.mock_outputs_arr)
        
        # Attributes from Plotter (grandparent) via DRP's super() call
        self.assertIs(plotter.model_wrapper, self.mock_diffusion_mw)
        self.assertEqual(plotter.ncols, expected_ncols)
        self.assertEqual(plotter.nrows, expected_nrows)
        
        # Check thumbnail_size modification
        # Default from Plotter is (4,4). DRP does: (super().thumbnail_size[0] * 2, super().thumbnail_size[1])
        # So, (4*2, 4) = (8,4)
        self.assertEqual(plotter.thumbnail_size, (8, 4))


    @patch.object(DiffusionRandomPlotter, 'imshow') # Mock from Plotter
    @patch.object(DiffusionRandomPlotter, 'display', new_callable=MagicMock) # Mock LazyProperty
    def test_call_method(self, mock_display_prop, mock_imshow_method):
        plotter = self.create_plotter() # nrows is 5
        
        # Mock model_wrapper.generate()
        # It should return a list/array of images, where number of images = self.nrows
        mock_generated_images_list = [MagicMock(name=f"GeneratedImg_{i}") for i in range(plotter.nrows)]
        self.mock_diffusion_mw.generate.return_value = mock_generated_images_list
        
        # Call the method
        returned_display = plotter()

        # 1. Check model_wrapper.generate call
        self.mock_diffusion_mw.generate.assert_called_once_with(plotter.nrows, return_steps=True)

        # 2. Check self.imshow calls for each image from generate
        expected_imshow_calls = []
        for i, img_mock in enumerate(mock_generated_images_list):
            expected_imshow_calls.append(call(i, img_mock))
        mock_imshow_method.assert_has_calls(expected_imshow_calls)
        self.assertEqual(mock_imshow_method.call_count, len(mock_generated_images_list))

        # 3. Check if display is returned
        self.assertIs(returned_display, mock_display_prop)
        
        # 4. Check reset_display decorator effect
        self.assertIsNotNone(plotter._display) # Accessed by __call__


if __name__ == '__main__':
    unittest.main()
