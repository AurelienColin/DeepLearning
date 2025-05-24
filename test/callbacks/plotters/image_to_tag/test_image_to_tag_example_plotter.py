import unittest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

# Mock Rignak (used by parent Plotter)
mock_rignak_display_i2t_ep = MagicMock()
sys.modules['Rignak.custom_display'] = mock_rignak_display_i2t_ep
mock_rignak_lazy_property_i2t_ep = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property_i2t_ep

# Mock ModelWrapper and OutputSpace
mock_model_wrapper_module_i2t_ep = MagicMock()
sys.modules['src.models.model_wrapper'] = mock_model_wrapper_module_i2t_ep
mock_output_space_module_i2t_ep = MagicMock()
sys.modules['src.output_spaces.output_space'] = mock_output_space_module_i2t_ep

# Mock numpy
mock_numpy_i2t_ep = MagicMock()
sys.modules['numpy'] = mock_numpy_i2t_ep

# Mock losses (cross_entropy, one_minus_dice)
mock_losses_module_i2t_ep = MagicMock()
sys.modules['src.losses.losses'] = mock_losses_module_i2t_ep

# Mock the external plot function
mock_benchmark_utils_plot_i2t_ep = MagicMock()
sys.modules['src.trainers.image_to_image_trainers.run.benchmark.utils'] = MagicMock(
    plot=mock_benchmark_utils_plot_i2t_ep
)

# Import after mocks
from callbacks.plotters.image_to_tag.image_to_tag_example_plotter import ImageToTagExamplePlotter
# PlotterFromArrays is parent, Plotter is grandparent

class TestImageToTagExamplePlotter(unittest.TestCase):

    def setUp(self):
        mock_rignak_display_i2t_ep.Display.reset_mock()
        mock_rignak_lazy_property_i2t_ep.LazyProperty.reset_mock() # Reset decorator if it has state
        mock_model_wrapper_module_i2t_ep.ModelWrapper.reset_mock()
        mock_output_space_module_i2t_ep.OutputSpace.reset_mock()
        mock_numpy_i2t_ep.reset_mock()
        mock_losses_module_i2t_ep.cross_entropy.reset_mock()
        mock_losses_module_i2t_ep.one_minus_dice.reset_mock()
        mock_benchmark_utils_plot_i2t_ep.reset_mock()

        self.mock_mw = MagicMock(spec=mock_model_wrapper_module_i2t_ep.ModelWrapper)
        self.mock_keras_model = MagicMock(name="KerasModelInstance")
        self.mock_mw.model = self.mock_keras_model
        
        self.mock_output_space = MagicMock(spec=mock_output_space_module_i2t_ep.OutputSpace)
        self.mock_output_space.n = 20 # Total number of possible tags
        self.mock_output_space.tag_names = [f"tag_{j}" for j in range(self.mock_output_space.n)]

        self.num_samples_original = 12 # More than ncols=10 to test slicing
        self.num_tags_original = self.mock_output_space.n 
        
        self.mock_inputs_arr = MagicMock(name="inputs_array_i2t")
        self.mock_inputs_arr.shape = [self.num_samples_original, 32, 32, 3]
        
        self.mock_outputs_arr = MagicMock(name="outputs_array_i2t") # Binary matrix
        self.mock_outputs_arr.shape = [self.num_samples_original, self.num_tags_original]

        # Mock slicing for inputs/outputs in __init__
        self.ncols_expected = min(self.num_samples_original, 10)
        self.mock_inputs_sliced = MagicMock(name="inputs_sliced_i2t")
        self.mock_inputs_sliced.shape = [self.ncols_expected, 32, 32, 3]
        self.mock_inputs_arr.__getitem__.return_value = self.mock_inputs_sliced
        
        self.mock_outputs_sliced = MagicMock(name="outputs_sliced_i2t")
        self.mock_outputs_sliced.shape = [self.ncols_expected, self.num_tags_original]
        self.mock_outputs_arr.__getitem__.return_value = self.mock_outputs_sliced


    def create_plotter(self):
        return ImageToTagExamplePlotter(
            inputs=self.mock_inputs_arr,
            outputs=self.mock_outputs_arr,
            model_wrapper=self.mock_mw,
            output_space=self.mock_output_space
        )

    def test_initialization(self):
        plotter = self.create_plotter()

        expected_ncols = self.ncols_expected
        expected_nrows = 3
        expected_thumb_size = (6,5)
        expected_max_tags = min(10, self.mock_outputs_sliced.shape[1])

        self.mock_inputs_arr.__getitem__.assert_called_once_with(slice(None, expected_ncols))
        self.mock_outputs_arr.__getitem__.assert_called_once_with(slice(None, expected_ncols))

        self.assertIs(plotter.inputs, self.mock_inputs_sliced)
        self.assertIs(plotter.outputs, self.mock_outputs_sliced)
        self.assertIs(plotter.model_wrapper, self.mock_mw)
        self.assertIs(plotter.output_space, self.mock_output_space)
        self.assertEqual(plotter.ncols, expected_ncols)
        self.assertEqual(plotter.nrows, expected_nrows)
        self.assertEqual(plotter.thumbnail_size, expected_thumb_size)
        self.assertEqual(plotter.max_tags, expected_max_tags)
        self.assertIsNone(plotter.logs)
        self.assertIsNone(plotter._indices) # Lazy prop

    def test_indices_lazy_property(self):
        plotter = self.create_plotter()
        plotter.max_tags = 3 # For predictable testing
        
        # Mock plotter.outputs (which is self.mock_outputs_sliced)
        # Sample 0: 2 true tags, needs 1 random
        # Sample 1: 4 true tags, gets truncated to 3
        mock_output_sample0 = MagicMock(name="output_s0")
        mock_argwhere_s0 = MagicMock(name="argwhere_s0_result"); mock_argwhere_s0.shape=[2,1] # 2 true tags
        mock_output_sample0.__eq__.return_value = MagicMock() # For (output == 1)
        mock_numpy_i2t_ep.argwhere.side_effect = [mock_argwhere_s0, MagicMock()] # For s0, s1
        mock_argwhere_s0.__getitem__.return_value = [0,1] # Indices of true tags for sample 0 ([0,1])

        mock_output_sample1 = MagicMock(name="output_s1")
        mock_argwhere_s1 = MagicMock(name="argwhere_s1_result"); mock_argwhere_s1.shape=[4,1] # 4 true tags
        mock_output_sample1.__eq__.return_value = MagicMock()
        mock_argwhere_s1.__getitem__.return_value = [2,3,4,5] # Indices of true tags for sample 1

        plotter.outputs = [mock_output_sample0, mock_output_sample1] # Override with detailed mocks
        
        # Mock np.random.choice for sample 0
        mock_random_choice_s0 = MagicMock(name="random_choice_s0_result")
        mock_numpy_i2t_ep.random.choice.return_value = mock_random_choice_s0
        mock_random_choice_s0.tolist.return_value = [10] # Randomly chosen index

        # Mock np.concatenate for sample 0
        mock_concat_s0 = MagicMock(name="concat_s0_result")
        mock_numpy_i2t_ep.concatenate.return_value = mock_concat_s0
        mock_concat_s0.tolist.return_value = [0,1,10] # After sort, it should be [0,1,10]

        # Mock np.array to finalize
        mock_final_indices_array = MagicMock(name="final_indices_array")
        mock_numpy_i2t_ep.array.return_value = mock_final_indices_array

        indices_val = plotter.indices

        self.assertEqual(mock_numpy_i2t_ep.argwhere.call_count, 2)
        mock_numpy_i2t_ep.random.choice.assert_called_once() # Only for sample 0
        # Check args for random.choice for sample 0:
        # remaining_indices = [i for i in range(20) if i not in [0,1]]
        # self.max_tags - truth_indices.shape[0] = 3 - 2 = 1
        expected_remaining_s0 = [i for i in range(self.mock_output_space.n) if i not in [0,1]]
        mock_numpy_i2t_ep.random.choice.assert_called_with(expected_remaining_s0, 1, replace=False)
        
        mock_numpy_i2t_ep.concatenate.assert_called_once_with(([0,1], mock_random_choice_s0))

        # Final np.array call with the list of processed indices for each sample
        # Expected for sample 0: sorted([0,1,10]) -> [0,1,10]
        # Expected for sample 1: sorted([2,3,4,5][:3]) -> sorted([2,3,4]) -> [2,3,4]
        mock_numpy_i2t_ep.array.assert_called_once_with([[0,1,10], [2,3,4]])
        self.assertIs(indices_val, mock_final_indices_array)
        self.assertIs(plotter.indices, indices_val) # Cached

    def test_get_labels(self):
        plotter = self.create_plotter()
        test_indices = [1, 5, 0]
        # self.mock_output_space.tag_names = ["tag_0", "tag_1", ..., "tag_5", ...]
        expected_labels = ["tag_1", "tag_5", "tag_0"]
        
        labels = plotter.get_labels(test_indices)
        self.assertEqual(labels, expected_labels)

    @patch.object(ImageToTagExamplePlotter, 'imshow')
    def test_call_for_inputs(self, mock_imshow_method):
        plotter = self.create_plotter()
        # plotter.inputs is self.mock_inputs_sliced
        # Make it iterable for the test
        mock_input_images = [MagicMock(name=f"InputImg{i}") for i in range(self.ncols_expected)]
        plotter.inputs = mock_input_images

        plotter.call_for_inputs()
        
        expected_calls = [call(i, img, interpolation="bicubic") for i, img in enumerate(mock_input_images)]
        mock_imshow_method.assert_has_calls(expected_calls)

    @patch.object(ImageToTagExamplePlotter, 'get_labels')
    @patch.object(ImageToTagExamplePlotter, 'display', new_callable=MagicMock)
    def test_call_for_predictions(self, mock_display_prop, mock_get_labels_method):
        plotter = self.create_plotter()
        # Setup mocks for data used in this method
        num_displayed_samples = self.ncols_expected
        plotter.indices = [[0,1]] * num_displayed_samples # Mocked indices for each sample
        plotter.outputs = [[0.0,1.0]] * num_displayed_samples # Mocked outputs
        # logs[-1] are the latest predictions
        plotter.logs = [[ [0.1, 0.9] ] * num_displayed_samples] # Mocked predictions (1 "epoch", N samples, M tags)

        mock_losses_module_i2t_ep.one_minus_dice.return_value.numpy.return_value = 0.2 # 1 - 0.2 = 0.8 Dice
        mock_losses_module_i2t_ep.cross_entropy.return_value.numpy.return_value = 0.5 # CE

        mock_get_labels_method.return_value = ["label0", "label1"] # Mocked labels
        
        mock_subplot = MagicMock()
        mock_subplot.ax = MagicMock() # For ax.set_xticklabels
        mock_display_prop.__getitem__.return_value = mock_subplot

        plotter.call_for_predictions()

        self.assertEqual(mock_losses_module_i2t_ep.one_minus_dice.call_count, num_displayed_samples)
        self.assertEqual(mock_losses_module_i2t_ep.cross_entropy.call_count, num_displayed_samples)
        self.assertEqual(mock_get_labels_method.call_count, num_displayed_samples)
        self.assertEqual(mock_display_prop.__getitem__.call_count, num_displayed_samples) # One subplot per sample
        self.assertEqual(mock_subplot.barh.call_count, num_displayed_samples * 2) # output and prediction bars
        self.assertEqual(mock_subplot.ax.set_xticklabels.call_count, num_displayed_samples)


    # Test for call_for_logs is complex due to external plot. We'll check args.
    def test_call_for_logs(self):
        plotter = self.create_plotter()
        num_displayed_samples = self.ncols_expected
        num_tags_per_sample = 2
        plotter.indices = [[0,1]] * num_displayed_samples # Mocked indices (num_tags_per_sample tags)
        
        # logs shape: (num_epochs, num_samples, num_total_tags)
        # Here, self.logs has one "epoch" of data after __call__ runs once.
        # Let's assume logs has 2 epochs for this test.
        mock_logs_data = MagicMock(name="logs_data_arr")
        # Sliced logs: self.logs[:, i, indices]
        mock_sliced_log_for_plot = MagicMock(name="sliced_log_for_plot")
        mock_logs_data.__getitem__.return_value = mock_sliced_log_for_plot
        plotter.logs = mock_logs_data

        mock_numpy_i2t_ep.clip.return_value = mock_sliced_log_for_plot # clip returns the same mock

        mock_get_labels_method = MagicMock(return_value=["label0", "label1"])
        plotter.get_labels = mock_get_labels_method # Patch instance method

        plotter.call_for_logs()
        
        limit = 1e-3
        mock_numpy_i2t_ep.clip.assert_called_with(ANY, limit, 1-limit) # ANY is plotter.logs[:, i, indices]
        
        self.assertEqual(mock_benchmark_utils_plot_i2t_ep.call_count, num_displayed_samples)
        # Check one call to the external plot function
        mock_benchmark_utils_plot_i2t_ep.assert_any_call(
            None, # x is None
            mock_sliced_log_for_plot, # y data
            ylabel="Prediction",
            xlabel="Epochs",
            labels=["label0", "label1"],
            xmin=0,
            ymin=limit / 2,
            ymax=1 - limit / 2,
            yscale="logit"
        )


    @patch.object(ImageToTagExamplePlotter, 'call_for_inputs')
    @patch.object(ImageToTagExamplePlotter, 'call_for_predictions')
    @patch.object(ImageToTagExamplePlotter, 'call_for_logs')
    def test_call_method_flow(self, mock_call_logs, mock_call_preds, mock_call_inputs):
        plotter = self.create_plotter()
        
        mock_preds_raw = MagicMock(name="preds_raw_tensor")
        mock_preds_numpy = MagicMock(name="preds_numpy_array")
        mock_preds_raw.numpy.return_value = mock_preds_numpy
        self.mock_keras_model.return_value = mock_preds_raw # model(inputs)

        # Case 1: self.logs is None (first call)
        plotter.logs = None
        mock_preds_numpy_newaxis = MagicMock(name="preds_newaxis")
        mock_preds_numpy.__getitem__.return_value = mock_preds_numpy_newaxis # For [np.newaxis]

        plotter()

        self.mock_keras_model.assert_called_once_with(plotter.inputs, training=False)
        mock_preds_numpy.__getitem__.assert_called_once_with(mock_numpy_i2t_ep.newaxis)
        self.assertIs(plotter.logs, mock_preds_numpy_newaxis)
        
        mock_call_inputs.assert_called_once()
        mock_call_preds.assert_called_once()
        mock_call_logs.assert_called_once()

        # Case 2: self.logs exists (subsequent call)
        mock_call_inputs.reset_mock(); mock_call_preds.reset_mock(); mock_call_logs.reset_mock()
        self.mock_keras_model.reset_mock(); mock_preds_numpy.__getitem__.reset_mock()
        
        existing_logs = MagicMock(name="existing_logs")
        plotter.logs = existing_logs
        mock_concatenated_logs = MagicMock(name="concatenated_logs")
        mock_numpy_i2t_ep.concatenate.return_value = mock_concatenated_logs
        
        plotter() # Call again

        self.mock_keras_model.assert_called_once_with(plotter.inputs, training=False)
        # preds[np.newaxis] happens again before concatenate
        mock_preds_numpy.__getitem__.assert_called_once_with(mock_numpy_i2t_ep.newaxis) 
        mock_numpy_i2t_ep.concatenate.assert_called_once_with((existing_logs, mock_preds_numpy_newaxis))
        self.assertIs(plotter.logs, mock_concatenated_logs)

        mock_call_inputs.assert_called_once()
        mock_call_preds.assert_called_once()
        mock_call_logs.assert_called_once()


if __name__ == '__main__':
    unittest.main()
