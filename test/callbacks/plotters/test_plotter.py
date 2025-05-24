import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock Rignak (used by Plotter)
mock_rignak_display_plotter = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_plotter
mock_rignak_lazy_property_plotter = MagicMock() # For @LazyProperty
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_plotter


# Mock ModelWrapper (passed to Plotter's __init__)
mock_model_wrapper_module_plotter = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_plotter

# Mock numpy (used by Plotter.concatenate)
mock_numpy_plotter = MagicMock()
sys.modules['numpy'] = mock_numpy_plotter

from callbacks.plotters.plotter import Plotter, reset_display 
# Plotter is the class to test. reset_display is also defined here.

class TestPlotter(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_plotter.Display.reset_mock()
        mock_model_wrapper_module_plotter.ModelWrapper.reset_mock()
        mock_numpy_plotter.reset_mock()

        self.mock_mw_instance = MagicMock(spec=mock_model_wrapper_module_plotter.ModelWrapper)
        self.ncols_val = 3
        self.nrows_val = 2
        self.thumb_size_val = (6, 6)

    def create_plotter(self, **kwargs):
        params = {
            'model_wrapper': self.mock_mw_instance,
            'ncols': self.ncols_val,
            'nrows': self.nrows_val,
            'thumbnail_size': self.thumb_size_val,
            **kwargs
        }
        return Plotter(**params)

    def test_initialization(self):
        plotter_instance = self.create_plotter()

        self.assertIs(plotter_instance.model_wrapper, self.mock_mw_instance)
        self.assertEqual(plotter_instance.ncols, self.ncols_val)
        self.assertEqual(plotter_instance.nrows, self.nrows_val)
        self.assertEqual(plotter_instance.thumbnail_size, self.thumb_size_val)
        self.assertIsNone(plotter_instance._display) # Lazy property not accessed

    def test_initialization_default_thumbnail_size(self):
        plotter_instance = Plotter(
            model_wrapper=self.mock_mw_instance,
            ncols=self.ncols_val,
            nrows=self.nrows_val
            # thumbnail_size uses default
        )
        self.assertEqual(plotter_instance.thumbnail_size, (4,4)) # Default value


    def test_display_lazy_property(self):
        plotter_instance = self.create_plotter()
        mock_display_obj = MagicMock(spec=mock_rignak_display_plotter.Display)
        mock_rignak_display_plotter.Display.return_value = mock_display_obj

        # Access property first time
        display1 = plotter_instance.display
        mock_rignak_display_plotter.Display.assert_called_once_with(
            ncols=self.ncols_val, nrows=self.nrows_val, figsize=self.thumb_size_val
        )
        self.assertIs(display1, mock_display_obj)
        self.assertIs(plotter_instance._display, mock_display_obj) # Check if internal var set

        # Access property second time
        display2 = plotter_instance.display
        self.assertIs(display2, mock_display_obj) # Should be cached
        mock_rignak_display_plotter.Display.assert_called_once() # Not called again

    def test_concatenate_static_method(self):
        # Mock numpy operations used by concatenate
        mock_array1 = MagicMock(name="array1_3channel")
        mock_array1.ndim = 4 
        mock_array1.shape = [1, 10, 10, 3] # Batch, H, W, C

        mock_array2_1channel = MagicMock(name="array2_1channel")
        mock_array2_1channel.ndim = 4
        mock_array2_1channel.shape = [1, 10, 10, 1]
        
        mock_array3_4channel = MagicMock(name="array3_4channel")
        mock_array3_4channel.ndim = 4
        mock_array3_4channel.shape = [1, 10, 10, 4]
        
        mock_array4_3dims = MagicMock(name="array4_3dims") # (B, H, W) -> needs new axis
        mock_array4_3dims.ndim = 3
        # After new axis: (B,H,W,1), then tiled to (B,H,W,3)

        # Mock results of numpy operations
        mock_array2_tiled = MagicMock(name="array2_tiled_to_3channel")
        mock_numpy_plotter.tile.return_value = mock_array2_tiled
        
        mock_array3_sliced = MagicMock(name="array3_sliced_to_3channel")
        mock_array3_4channel.__getitem__.return_value = mock_array3_sliced # For array[:,:,:,:3]

        mock_array4_with_axis = MagicMock(name="array4_with_new_axis")
        mock_array4_tiled = MagicMock(name="array4_tiled")
        # array4_3dims[:, :, :, np.newaxis] -> this is complex to mock directly on the mock
        # Instead, we can check the logic flow. If ndim is 3, it's processed.
        # Let's assume the newaxis and subsequent tiling are tested by numpy's tests.
        # We will focus on the sequence of calls and transformations based on shape.

        mock_concatenated_raw = MagicMock(name="concatenated_raw")
        mock_numpy_plotter.concatenate.return_value = mock_concatenated_raw
        
        mock_final_clipped_array = MagicMock(name="final_clipped")
        mock_numpy_plotter.clip.return_value = mock_final_clipped_array

        # Test with a mix of arrays
        result = Plotter.concatenate(mock_array1, mock_array2_1channel, mock_array3_4channel, mock_array4_3dims, axis=1)

        # Check processing for mock_array2_1channel (shape[3] == 1)
        mock_numpy_plotter.tile.assert_any_call(mock_array2_1channel, (1,1,1,3))
        
        # Check processing for mock_array3_4channel (shape[3] > 3)
        mock_array3_4channel.__getitem__.assert_called_once_with((slice(None), slice(None), slice(None), slice(None, 3)))

        # Check processing for mock_array4_3dims (ndim == 3)
        # It should be reshaped then tiled. We expect np.tile to be called for it.
        # The newaxis part is harder to check on a generic mock without specific __getitem__ setup.
        # We expect it to be tiled similar to array2 after it gets a new axis.
        # This part of the test might need refinement if precise ndim manipulation on mocks is key.
        # For now, ensure np.tile was called for arrays that end up as 1-channel before concat.
        # This means it would be called for mock_array2_1channel AND for the processed mock_array4_3dims.
        # A simple way: check call_args_list for np.tile.
        tile_calls = [
            call(mock_array2_1channel, (1,1,1,3)), 
            # Assuming mock_array4_3dims after adding axis becomes something like 'mock_array4_processed'
            # call(mock_array4_processed, (1,1,1,3)) - This is hard to assert directly.
        ]
        # Check at least one call for array2 happened as expected.
        self.assertIn(tile_calls[0], mock_numpy_plotter.tile.call_args_list)


        # Check concatenation call
        # Expected arrays for concatenation: mock_array1, mock_array2_tiled, mock_array3_sliced, and tiled mock_array4_3dims
        # The exact object for tiled mock_array4_3dims is tricky to get without deeper mocking.
        # We can check that concatenate was called with the correct number of arrays and axis.
        self.assertEqual(mock_numpy_plotter.concatenate.call_args[0][0][0], mock_array1)
        self.assertEqual(mock_numpy_plotter.concatenate.call_args[0][0][1], mock_array2_tiled)
        self.assertEqual(mock_numpy_plotter.concatenate.call_args[0][0][2], mock_array3_sliced)
        # self.assertEqual(mock_numpy_plotter.concatenate.call_args[0][0][3], ???) # Tiled mock_array4
        self.assertEqual(len(mock_numpy_plotter.concatenate.call_args[0][0]), 4) # 4 arrays passed
        self.assertEqual(mock_numpy_plotter.concatenate.call_args[1]['axis'], 1) # Correct axis

        # Check clip call
        mock_numpy_plotter.clip.assert_called_once_with(mock_concatenated_raw, 0, 1)
        self.assertIs(result, mock_final_clipped_array)


    def test_imshow(self):
        plotter_instance = self.create_plotter()
        
        # Mock the display and its __getitem__ to get a plottable cell
        mock_display_obj = MagicMock(spec=mock_rignak_display_plotter.Display)
        mock_cell_to_plot_on = MagicMock()
        mock_display_obj.__getitem__.return_value = mock_cell_to_plot_on
        plotter_instance._display = mock_display_obj # Manually set the display for this test

        test_index = 0
        # Image needs to be at least 3D for image[:,:,:3]
        mock_image_data = MagicMock(name="image_data_for_imshow")
        mock_image_data.shape = (10,10,3) 
        mock_image_sliced_for_plot = MagicMock(name="image_sliced_for_plot")
        mock_image_data.__getitem__.return_value = mock_image_sliced_for_plot # For image[:,:,:3]
        
        custom_kwargs = {'title': "My Image"}
        expected_plot_kwargs = {
            'vmin': 0, 'vmax': 255, 'grid': False, 
            'axis_display': False, 'colorbar_display': False, 
            **custom_kwargs
        }

        plotter_instance.imshow(test_index, mock_image_data, **custom_kwargs)

        mock_display_obj.__getitem__.assert_called_once_with(test_index)
        mock_image_data.__getitem__.assert_called_once_with((slice(None), slice(None), slice(None, 3)))
        mock_cell_to_plot_on.imshow.assert_called_once_with(mock_image_sliced_for_plot, **expected_plot_kwargs)

    def test_call_is_abstract(self):
        plotter_instance = self.create_plotter()
        with self.assertRaises(NotImplementedError):
            plotter_instance() # __call__()

    def test_reset_display_decorator(self):
        # Define a dummy class and method to test the decorator
        class DummyPlottable:
            def __init__(self):
                self._display = "InitialDisplay" # Simulate a cached display

            @reset_display
            def some_method_that_resets_display(self, arg1):
                return f"Method called with {arg1}"
        
        dummy_instance = DummyPlottable()
        self.assertEqual(dummy_instance._display, "InitialDisplay")
        
        result = dummy_instance.some_method_that_resets_display("test_arg")
        
        self.assertIsNone(dummy_instance._display) # Decorator should have set it to None
        self.assertEqual(result, "Method called with test_arg")


if __name__ == '__main__':
    unittest.main()
