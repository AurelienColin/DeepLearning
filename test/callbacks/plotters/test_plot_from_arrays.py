import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock Rignak (used by parent Plotter)
mock_rignak_display_pfa = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_pfa
mock_rignak_lazy_property_pfa = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_pfa

# Mock ModelWrapper (passed to grandparent Plotter's __init__)
mock_model_wrapper_module_pfa = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_pfa

# Mock numpy (used by parent Plotter and for type hints here)
# We need np.ndarray for type hints, but can mock the module itself.
mock_numpy_pfa = MagicMock()
# Allow np.ndarray to be used as a type hint without error, even if numpy is fully mocked.
# This is a bit of a hack; for type checking, it's usually better to have the actual type.
# However, for runtime mocking, this prevents `AttributeError` if `np.ndarray` is accessed.
mock_numpy_pfa.ndarray = type(MagicMock(name="ndarray_type_mock")())
sys.modules['numpy'] = mock_numpy_pfa


from callbacks.plotters.plot_from_arrays import PlotterFromArrays
# Plotter is the parent class.

class TestPlotterFromArrays(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_pfa.Display.reset_mock()
        mock_model_wrapper_module_pfa.ModelWrapper.reset_mock()
        mock_numpy_pfa.reset_mock() # Reset the module mock

        self.mock_mw_instance_pfa = MagicMock(spec=mock_model_wrapper_module_pfa.ModelWrapper)
        self.mock_inputs_array = MagicMock(spec=mock_numpy_pfa.ndarray) # Use mocked ndarray for spec
        self.mock_outputs_array = MagicMock(spec=mock_numpy_pfa.ndarray)
        
        self.ncols_val_pfa = 2
        self.nrows_val_pfa = 2

    def create_plotter_from_arrays(self, **kwargs_plotter):
        # Parent Plotter needs model_wrapper, ncols, nrows
        # PlotterFromArrays needs inputs, outputs
        params_plotter = {
            'model_wrapper': self.mock_mw_instance_pfa,
            'ncols': self.ncols_val_pfa,
            'nrows': self.nrows_val_pfa,
            **kwargs_plotter.pop('plotter_kwargs', {})
        }
        
        # Create a dummy concrete class for instantiation
        class ConcretePlotterFromArrays(PlotterFromArrays):
            def __call__(self):
                pass # Dummy implementation

        instance = ConcretePlotterFromArrays(
            inputs=self.mock_inputs_array,
            outputs=self.mock_outputs_array,
            **params_plotter
        )
        return instance

    def test_initialization(self):
        plotter_instance = self.create_plotter_from_arrays()

        # Attributes from PlotterFromArrays
        self.assertIs(plotter_instance.inputs, self.mock_inputs_array)
        self.assertIs(plotter_instance.outputs, self.mock_outputs_array)

        # Attributes from parent Plotter
        self.assertIs(plotter_instance.model_wrapper, self.mock_mw_instance_pfa)
        self.assertEqual(plotter_instance.ncols, self.ncols_val_pfa)
        self.assertEqual(plotter_instance.nrows, self.nrows_val_pfa)
        self.assertEqual(plotter_instance.thumbnail_size, (4,4)) # Default from Plotter


    # Test for plotting mechanism:
    # `PlotterFromArrays` itself does not implement a `plot` or `__call__` method.
    # This responsibility lies with its concrete subclasses.
    # We can test that `Plotter.__call__` is still abstract if not implemented.
    def test_call_is_abstract_via_parent(self):
        class AbstractIshPlotterFromArrays(PlotterFromArrays):
            # No __call__ implementation
            pass

        plotter_instance_abstract = AbstractIshPlotterFromArrays(
            inputs=self.mock_inputs_array,
            outputs=self.mock_outputs_array,
            model_wrapper=self.mock_mw_instance_pfa,
            ncols=1, nrows=1
        )
        with self.assertRaises(NotImplementedError):
            plotter_instance_abstract()

if __name__ == '__main__':
    unittest.main()
