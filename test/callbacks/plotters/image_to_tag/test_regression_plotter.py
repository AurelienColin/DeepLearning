import unittest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

# Mock Rignak (used by parent Plotter)
mock_rignak_display_rp = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_rp
mock_rignak_lazy_property_rp = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_rp

# Mock ModelWrapper and ClassificationGenerator (and its OutputSpace)
mock_model_wrapper_module_rp = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_rp
mock_classification_generator_module_rp = MagicMock()
sys.modules['src.generators.image_to_tag.classification_generator'] = mock_classification_generator_module_rp

# Mock numpy and scipy.stats
mock_numpy_rp = MagicMock()
sys.modules['numpy'] = mock_numpy_rp
mock_scipy_stats_rp = MagicMock()
sys.modules['scipy.stats'] = mock_scipy_stats_rp

# Import after mocks
from callbacks.plotters.image_to_tag.regression_plotter import RegressionPlotter
# PlotterFromGenerator is parent, Plotter is grandparent.

class TestRegressionPlotter(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_rp.Display.reset_mock()
        mock_model_wrapper_module_rp.ModelWrapper.reset_mock()
        mock_classification_generator_module_rp.ClassificationGenerator.reset_mock()
        mock_numpy_rp.reset_mock()
        mock_scipy_stats_rp.pearsonr.reset_mock()


        self.mock_mw = MagicMock(spec=mock_model_wrapper_module_rp.ModelWrapper)
        self.mock_keras_model = MagicMock(name="KerasModelInstance")
        self.mock_mw.model = self.mock_keras_model
        
        self.mock_generator = MagicMock(spec=mock_classification_generator_module_rp.ClassificationGenerator)
        self.mock_output_space = MagicMock(name="OutputSpaceInstance")
        self.num_output_dims = 2 # Example: 2 regression outputs (e.g., x and y coordinates)
        self.mock_output_space.n = self.num_output_dims
        self.mock_output_space.tag_names = [f"dim_{j}" for j in range(self.num_output_dims)]
        self.mock_generator.output_space = self.mock_output_space
        self.mock_generator.batch_size = 4 # Example batch size
        
        self.steps_val = 3 # Number of steps to run generator

    def create_plotter(self):
        return RegressionPlotter(
            generator=self.mock_generator,
            steps=self.steps_val,
            model_wrapper=self.mock_mw
        )

    def test_initialization(self):
        plotter = self.create_plotter()

        self.assertIs(plotter.generator, self.mock_generator)
        self.assertEqual(plotter.steps, self.steps_val)
        self.assertIs(plotter.model_wrapper, self.mock_mw)
        
        # Overridden by RegressionPlotter
        self.assertEqual(plotter.ncols, self.num_output_dims) 
        self.assertEqual(plotter.nrows, 1)
        self.assertEqual(plotter.thumbnail_size, (10,10))


    @patch.object(RegressionPlotter, 'display', new_callable=MagicMock)
    def test_call_method_data_aggregation_and_plotting(self, mock_display_prop):
        plotter = self.create_plotter()
        
        # Mock data from generator (batch_size * steps samples in total)
        total_samples = self.mock_generator.batch_size * self.steps_val
        # Each sample has num_output_dims regression values
        
        # Mock generator __next__ to return (inputs, outputs)
        # Outputs shape: (batch_size, num_output_dims)
        # Predictions shape: (batch_size, num_output_dims)
        mock_outputs_batches = [
            MagicMock(name=f"outputs_b{i}") for i in range(self.steps_val)
        ]
        mock_preds_numpy_batches = [
            MagicMock(name=f"preds_numpy_b{i}") for i in range(self.steps_val)
        ]
        
        def generator_next_side_effect():
            for i in range(self.steps_val):
                yield (MagicMock(name=f"inputs_b{i}"), mock_outputs_batches[i])
        self.mock_generator.__next__.side_effect = generator_next_side_effect()

        def model_predict_side_effect(inputs_arg, training):
            # Find which input batch this is to return corresponding pred batch
            # This is a bit fragile if inputs_arg isn't exactly what we expect.
            # For now, assume it cycles through predictions.
            idx = self.mock_keras_model.call_count -1 # 0-indexed call count
            raw_tensor = MagicMock(name=f"raw_preds_b{idx}")
            raw_tensor.numpy.return_value = mock_preds_numpy_batches[idx]
            return raw_tensor
        self.mock_keras_model.side_effect = model_predict_side_effect


        # Mock numpy.zeros for results array
        mock_results_zeros_array = MagicMock(name="results_zeros")
        # This array will be indexed like: results[k0:k1, :, 0] = outputs_batch
        # We need to mock its __setitem__ or assume it works like a numpy array.
        # For simplicity, we'll assume it gets populated and focus on how its slices are used.
        mock_numpy_rp.zeros.return_value = mock_results_zeros_array

        # Mock data slices from the 'results' array for each dim:
        # truth = results[:, i, 0], preds = results[:, i, 1]
        mock_truth_dim0 = MagicMock(name="truth_d0"); mock_truth_dim0.tolist.return_value = [1,2]
        mock_preds_dim0 = MagicMock(name="preds_d0"); mock_preds_dim0.tolist.return_value = [1.1, 1.9]
        mock_truth_dim1 = MagicMock(name="truth_d1"); mock_truth_dim1.tolist.return_value = [3,4]
        mock_preds_dim1 = MagicMock(name="preds_d1"); mock_preds_dim1.tolist.return_value = [3.1, 3.9]
        
        # Slicing results[:, i, 0] and results[:, i, 1]
        # This needs mock_results_zeros_array to handle __getitem__ correctly.
        def results_getitem_side_effect(key):
            # key will be (slice(None), dimension_index, truth_or_pred_index)
            dim_idx = key[1]
            type_idx = key[2]
            if dim_idx == 0 and type_idx == 0: return mock_truth_dim0
            if dim_idx == 0 and type_idx == 1: return mock_preds_dim0
            if dim_idx == 1 and type_idx == 0: return mock_truth_dim1
            if dim_idx == 1 and type_idx == 1: return mock_preds_dim1
            return MagicMock(name=f"slice_other_{dim_idx}_{type_idx}")
        mock_results_zeros_array.__getitem__.side_effect = results_getitem_side_effect


        # Mock np.min/max for vmin/vmax
        mock_numpy_rp.min.side_effect = lambda x: x.tolist()[0] # Return first element for simplicity
        mock_numpy_rp.max.side_effect = lambda x: x.tolist()[-1] # Return last element

        # Mock display[i].plot_regression
        mock_plot_regression_cell = MagicMock()
        mock_display_prop.__getitem__.return_value = mock_plot_regression_cell
        
        # Mock MAE: np.mean(np.abs(truth - preds))
        mock_abs_diff_mae = MagicMock(name="abs_diff_mae")
        mock_numpy_rp.abs.return_value = mock_abs_diff_mae
        mock_mean_mae = MagicMock(name="mean_mae_val") # Scalar value
        mock_numpy_rp.mean.return_value = mock_mean_mae
        
        # Mock pearsonr: pearsonr(truth, preds).statistic
        mock_pearson_result = MagicMock(name="PearsonResultTuple")
        mock_pearson_result.statistic = 0.95 # Example PCC
        mock_scipy_stats_rp.pearsonr.return_value = mock_pearson_result
        
        # Call the method
        returned_display, returned_logs = plotter()

        # 1. Check initialization of results array
        mock_numpy_rp.zeros.assert_called_once_with((total_samples, self.num_output_dims, 2))

        # 2. Check generator and model calls (steps times)
        self.assertEqual(self.mock_generator.__next__.call_count, self.steps_val)
        self.assertEqual(self.mock_keras_model.call_count, self.steps_val)

        # 3. Check population of results array (self.steps * 2 __setitem__ calls for outputs and preds)
        # This is hard to assert precisely without a stateful mock for results_zeros_array.
        # We trust that results[k0:k1, :, 0] = outputs and results[k0:k1, :, 1] = predictions happen.

        # 4. Loop for each output dimension (num_output_dims times)
        self.assertEqual(mock_display_prop.__getitem__.call_count, self.num_output_dims)
        self.assertEqual(mock_plot_regression_cell.plot_regression.call_count, self.num_output_dims)
        self.assertEqual(mock_numpy_rp.abs.call_count, self.num_output_dims) # For MAE
        self.assertEqual(mock_numpy_rp.mean.call_count, self.num_output_dims) # For MAE
        self.assertEqual(mock_scipy_stats_rp.pearsonr.call_count, self.num_output_dims)

        # Check one call to plot_regression (e.g., for first dimension)
        vmin_dim0 = mock_truth_dim0.tolist()[0]
        vmax_dim0 = mock_truth_dim0.tolist()[-1]
        mock_plot_regression_cell.plot_regression.assert_any_call(
            mock_truth_dim0, mock_preds_dim0, 
            ylabel="Predicted value", xlabel="True value", 
            xmin=vmin_dim0, ymin=vmin_dim0, xmax=vmax_dim0, ymax=vmax_dim0
        )
        
        # 5. Check logs dictionary content
        expected_logs = {
            'tag': [f"dim_{j}" for j in range(self.num_output_dims)],
            'mae': [mock_mean_mae] * self.num_output_dims, # Same mock_mean_mae due to simple mock_numpy_rp.mean
            'pcc': [mock_pearson_result.statistic] * self.num_output_dims
        }
        self.assertEqual(returned_logs, expected_logs)
        
        self.assertIs(returned_display, mock_display_prop)


if __name__ == '__main__':
    unittest.main()
