import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock Rignak (used by parent Plotter)
mock_rignak_display_pfg = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_pfg
mock_rignak_lazy_property_pfg = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_pfg

# Mock ModelWrapper (passed to grandparent Plotter's __init__)
mock_model_wrapper_module_pfg = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_pfg

# Mock ClassificationGenerator (passed to PlotterFromGenerator's __init__)
mock_classification_generator_module_pfg = MagicMock()
sys.modules['src.generators.image_to_tag.classification_generator'] = mock_classification_generator_module_pfg

# Mock numpy (used by parent Plotter)
mock_numpy_pfg = MagicMock()
sys.modules['numpy'] = mock_numpy_pfg

from callbacks.plotters.plotter_from_generator import PlotterFromGenerator
# Plotter is the parent class.

# Since PlotterFromGenerator is effectively abstract without a __call__ or plot method,
# we can create a dummy concrete subclass for testing purposes if needed,
# or test its __init__ directly.

class TestPlotterFromGenerator(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_pfg.Display.reset_mock()
        mock_model_wrapper_module_pfg.ModelWrapper.reset_mock()
        mock_classification_generator_module_pfg.ClassificationGenerator.reset_mock()
        mock_numpy_pfg.reset_mock()

        self.mock_mw_instance_pfg = MagicMock(spec=mock_model_wrapper_module_pfg.ModelWrapper)
        self.mock_generator_instance = MagicMock(spec=mock_classification_generator_module_pfg.ClassificationGenerator)
        
        self.ncols_val_pfg = 3
        self.nrows_val_pfg = 1
        self.steps_val_pfg = 5

    def create_plotter_from_generator(self, **kwargs_plotter):
        # Parent Plotter needs model_wrapper, ncols, nrows
        # PlotterFromGenerator needs generator, steps
        params_plotter = {
            'model_wrapper': self.mock_mw_instance_pfg,
            'ncols': self.ncols_val_pfg,
            'nrows': self.nrows_val_pfg,
            **kwargs_plotter.pop('plotter_kwargs', {}) # Pass specific plotter args if any
        }
        
        # Create a dummy concrete class that implements __call__ for Plotter parent
        # This allows us to instantiate PlotterFromGenerator for testing its __init__
        # without a NotImplementedError from Plotter.
        class ConcretePlotterFromGenerator(PlotterFromGenerator):
            def __call__(self):
                pass # Dummy implementation for instantiation

        instance = ConcretePlotterFromGenerator(
            generator=self.mock_generator_instance,
            steps=self.steps_val_pfg,
            **params_plotter # Pass parent Plotter's args
        )
        return instance

    def test_initialization(self):
        plotter_instance = self.create_plotter_from_generator()

        # Attributes from PlotterFromGenerator
        self.assertIs(plotter_instance.generator, self.mock_generator_instance)
        self.assertEqual(plotter_instance.steps, self.steps_val_pfg)

        # Attributes from parent Plotter
        self.assertIs(plotter_instance.model_wrapper, self.mock_mw_instance_pfg)
        self.assertEqual(plotter_instance.ncols, self.ncols_val_pfg)
        self.assertEqual(plotter_instance.nrows, self.nrows_val_pfg)
        self.assertEqual(plotter_instance.thumbnail_size, (4,4)) # Default from Plotter

    # Test for `get_batch` method:
    # The prompt asks to test `get_batch`. This method is not defined in the provided
    # `PlotterFromGenerator` code. If it were, a test would look like this:
    #
    # def test_get_batch_method(self):
    #     plotter_instance = self.create_plotter_from_generator()
    #     mock_batch_data = (MagicMock(name="batch_x"), MagicMock(name="batch_y"))
    #     self.mock_generator_instance.__next__.return_value = mock_batch_data # Or next(generator)
    #
    #     # Assuming get_batch was: def get_batch(self): return next(self.generator)
    #     # batch_x, batch_y = plotter_instance.get_batch()
    #
    #     # self.mock_generator_instance.__next__.assert_called_once()
    #     # self.assertIs(batch_x, mock_batch_data[0])
    #     # self.assertIs(batch_y, mock_batch_data[1])
    # As `get_batch` is not present, this test cannot be implemented for PlotterFromGenerator itself.
    # Concrete subclasses would be responsible for fetching and processing batches.

    # Test for `plot` method or `__call__`:
    # `Plotter.__call__` raises NotImplementedError. `PlotterFromGenerator` does not override it.
    # Thus, testing `__call__` on a direct instance of `PlotterFromGenerator` (if it weren't ABC now
    # due to Plotter's __call__) would just confirm the NotImplementedError.
    # The actual plotting logic will be in concrete subclasses.
    def test_call_is_abstract_via_parent(self):
        # Need to create a version of PlotterFromGenerator that *doesn't* have __call__ implemented
        # to test that Plotter's abstractness is inherited.
        
        # Temporarily remove __call__ from our ConcreteDouble for this specific test
        class AbstractIshPlotterFromGenerator(PlotterFromGenerator):
            # No __call__ implementation here
            pass

        plotter_instance_abstract = AbstractIshPlotterFromGenerator(
            generator=self.mock_generator_instance,
            steps=self.steps_val_pfg,
            model_wrapper=self.mock_mw_instance_pfg, # Required by Plotter
            ncols=1, nrows=1 # Required by Plotter
        )
        with self.assertRaises(NotImplementedError):
            plotter_instance_abstract()


if __name__ == '__main__':
    unittest.main()
