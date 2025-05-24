import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported by the modules under test
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.losses'] = mock_tf.keras.losses
sys.modules['tensorflow.keras.layers'] = mock_tf.keras.layers
sys.modules['tensorflow.keras.models'] = mock_tf.keras.models
sys.modules['tensorflow.keras.metrics'] = mock_tf.keras.metrics

# Mock Rignak
sys.modules['Rignak.lazy_property'] = MagicMock()

# Mock custom_objects (used in load_model)
sys.modules['src.modules.custom_objects'] = MagicMock(CUSTOM_OBJECTS={})

from losses.from_model.base_loss import LossFromModel

# Create a concrete implementation for testing
class ConcreteLossFromModel(LossFromModel):
    # Define input_shape in __init__ as LossFromModel expects it.
    def __init__(self, name="concrete_loss", input_shape=(32, 32, 3), reduction=mock_tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, input_shape=input_shape, reduction=reduction, **kwargs)
        # input_shape is used by the input_layer property, ensure it's set.
        self.input_shape = input_shape 


    @property
    def model_path(self) -> str:
        return "dummy/path/model.h5" # Needs to be implemented

    def call(self, y_true: mock_tf.Tensor, y_pred: mock_tf.Tensor) -> mock_tf.Tensor:
        # Dummy implementation for testing tracker methods
        # Example: return a mock tensor representing the difference
        return mock_tf.abs(y_true - y_pred)


class TestLossFromModel(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        # Reset the mock for CUSTOM_OBJECTS if it's modified by tests
        sys.modules['src.modules.custom_objects'].CUSTOM_OBJECTS = {}


    def test_initialization(self):
        name = "test_loss"
        input_shape = (64, 64, 3)
        loss_instance = ConcreteLossFromModel(name=name, input_shape=input_shape)

        self.assertEqual(loss_instance.name, name)
        self.assertEqual(loss_instance.input_shape, input_shape)
        self.assertIsNone(loss_instance._model) # Lazy property not accessed
        self.assertIsNone(loss_instance._input_layer) # Lazy property not accessed
        self.assertIsNone(loss_instance._tracker) # Lazy property not accessed
        # Check that the parent tf.keras.losses.Loss.__init__ was called
        mock_tf.keras.losses.Loss.assert_called_with(
            reduction=mock_tf.keras.losses.Reduction.AUTO, # Default from ConcreteLossFromModel
            name=name
        )


    @patch('tensorflow.keras.models.load_model')
    def test_model_property_loads_and_disables_trainable(self, mock_load_model):
        mock_loaded_model_instance = MagicMock(name="LoadedKerasModel")
        mock_load_model.return_value = mock_loaded_model_instance
        
        loss_instance = ConcreteLossFromModel()
        
        # Access the model property
        model = loss_instance.model 
        
        mock_load_model.assert_called_once_with(
            loss_instance.model_path, 
            compile=False, 
            custom_objects=sys.modules['src.modules.custom_objects'].CUSTOM_OBJECTS
        )
        self.assertIs(model, mock_loaded_model_instance)
        self.assertEqual(model.trainable, False) # Should be set to False after loading
        
        # Test caching
        self.assertIs(loss_instance.model, model)
        mock_load_model.assert_called_once() # Should only be called once


    def test_input_layer_property(self):
        input_shape = (32, 32, 1)
        loss_instance = ConcreteLossFromModel(input_shape=input_shape)
        
        mock_input_layer_instance = MagicMock(name="InputLayer")
        mock_tf.keras.layers.Input.return_value = mock_input_layer_instance

        # Access the input_layer property
        in_layer = loss_instance.input_layer

        mock_tf.keras.layers.Input.assert_called_once_with(shape=input_shape)
        self.assertIs(in_layer, mock_input_layer_instance)
        
        # Test caching
        self.assertIs(loss_instance.input_layer, in_layer)
        mock_tf.keras.layers.Input.assert_called_once()


    def test_tracker_property(self):
        name = "my_custom_loss"
        loss_instance = ConcreteLossFromModel(name=name, input_shape=(1,1,1)) # input_shape needed for init
        
        mock_mean_metric_instance = MagicMock(name="MeanMetric")
        mock_tf.keras.metrics.Mean.return_value = mock_mean_metric_instance

        # Access the tracker property
        tracker = loss_instance.tracker

        mock_tf.keras.metrics.Mean.assert_called_once_with(name=f"{name}_tracker")
        self.assertIs(tracker, mock_mean_metric_instance)

        # Test caching
        self.assertIs(loss_instance.tracker, tracker)
        mock_tf.keras.metrics.Mean.assert_called_once()

    def test_update_state_no_sample_weight(self):
        loss_instance = ConcreteLossFromModel()
        
        # Mock y_true, y_pred, and the return value of call()
        y_true_mock = MagicMock(name="y_true_tensor")
        y_pred_mock = MagicMock(name="y_pred_tensor")
        # The call method in ConcreteLossFromModel returns tf.abs(y_true - y_pred)
        # Let's mock what that would produce if tf was running
        mock_loss_value_tensor = MagicMock(name="CalculatedLossTensor")
        mock_tf.abs.return_value = mock_loss_value_tensor # Mocking the tf.abs call

        # Mock the tracker instance that would be created
        mock_tracker_instance = MagicMock(spec=mock_tf.keras.metrics.Mean)
        # Patch the 'tracker' property to return our mock_tracker_instance
        with patch.object(ConcreteLossFromModel, 'tracker', new_callable=PropertyMock, return_value=mock_tracker_instance):
            loss_instance.update_state(y_true_mock, y_pred_mock)

            # Check that the 'call' method was invoked (indirectly via self.call)
            # ConcreteLossFromModel.call uses tf.abs. So, tf.abs should have been called.
            mock_tf.abs.assert_called_once_with(y_true_mock - y_pred_mock)
            
            # Check that tracker.update_state was called with the result of 'call'
            mock_tracker_instance.update_state.assert_called_once_with(mock_loss_value_tensor)
            mock_tf.multiply.assert_not_called() # No sample weight

    def test_update_state_with_sample_weight(self):
        loss_instance = ConcreteLossFromModel()
        
        y_true_mock = MagicMock(name="y_true_sw")
        y_pred_mock = MagicMock(name="y_pred_sw")
        sample_weight_mock = MagicMock(name="sample_weight_val")
        mock_loss_value_tensor = MagicMock(name="LossTensorSW")
        mock_tf.abs.return_value = mock_loss_value_tensor # from call()

        mock_tf_sample_weight_tensor = MagicMock(name="SampleWeightTensor")
        mock_tf.convert_to_tensor.return_value = mock_tf_sample_weight_tensor
        
        mock_multiplied_loss_tensor = MagicMock(name="MultipliedLossTensor")
        mock_tf.multiply.return_value = mock_multiplied_loss_tensor

        mock_tracker_instance = MagicMock(spec=mock_tf.keras.metrics.Mean)
        with patch.object(ConcreteLossFromModel, 'tracker', new_callable=PropertyMock, return_value=mock_tracker_instance):
            loss_instance.update_state(y_true_mock, y_pred_mock, sample_weight=sample_weight_mock)

            mock_tf.abs.assert_called_once_with(y_true_mock - y_pred_mock)
            mock_tf.convert_to_tensor.assert_called_once_with(sample_weight_mock)
            mock_tf.multiply.assert_called_once_with(mock_loss_value_tensor, mock_tf_sample_weight_tensor)
            mock_tracker_instance.update_state.assert_called_once_with(mock_multiplied_loss_tensor)

    def test_result(self):
        loss_instance = ConcreteLossFromModel()
        mock_tracker_instance = MagicMock(spec=mock_tf.keras.metrics.Mean)
        mock_result_value = 0.5
        mock_tracker_instance.result.return_value = mock_result_value
        
        with patch.object(ConcreteLossFromModel, 'tracker', new_callable=PropertyMock, return_value=mock_tracker_instance):
            result = loss_instance.result()
            
            mock_tracker_instance.result.assert_called_once()
            self.assertEqual(result, mock_result_value)

    def test_reset_state(self):
        loss_instance = ConcreteLossFromModel()
        mock_tracker_instance = MagicMock(spec=mock_tf.keras.metrics.Mean)
        
        with patch.object(ConcreteLossFromModel, 'tracker', new_callable=PropertyMock, return_value=mock_tracker_instance):
            loss_instance.reset_state()
            
            mock_tracker_instance.reset_state.assert_called_once()

    def test_call_is_abstract(self):
        # Test that the base class's call method raises NotImplementedError
        # Need to instantiate LossFromModel directly, not ConcreteLossFromModel
        # However, LossFromModel itself is a tf.keras.losses.Loss, which might have its own __init__
        # that requires certain args.
        # The custom __init__ for LossFromModel takes name and input_shape.
        base_loss_instance = LossFromModel(name="base", input_shape=(1,1,1))
        with self.assertRaises(NotImplementedError):
            base_loss_instance.call(MagicMock(), MagicMock())
            
    def test_model_path_is_abstract(self):
        base_loss_instance = LossFromModel(name="base", input_shape=(1,1,1))
        with self.assertRaises(NotImplementedError):
            _ = base_loss_instance.model_path


if __name__ == '__main__':
    unittest.main()
