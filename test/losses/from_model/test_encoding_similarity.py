import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock TensorFlow before it's imported by the modules under test
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.backend'] = mock_tf.keras.backend # For K.mean, K.abs

# Mock Rignak (used by parent LossFromModel)
sys.modules['Rignak.lazy_property'] = MagicMock()

# Mock custom_objects (used by parent LossFromModel when loading model)
sys.modules['src.modules.custom_objects'] = MagicMock(CUSTOM_OBJECTS={})


from losses.from_model.encoding_similarity import EncodingSimilarity
# LossFromModel is the parent class, its testing is separate.

class TestEncodingSimilarity(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()

    def test_initialization(self):
        # EncodingSimilarity inherits __init__ from LossFromModel.
        name = "test_encoding_sim_loss"
        input_shape = (64, 64, 3) # Example input shape
        
        # We pass reduction to tf.keras.losses.Loss parent
        sim_loss = EncodingSimilarity(name=name, input_shape=input_shape, reduction="sum")

        self.assertEqual(sim_loss.name, name)
        self.assertEqual(sim_loss.input_shape, input_shape) # Set by LossFromModel's __init__
        # Check that parent tf.keras.losses.Loss.__init__ was called by LossFromModel's __init__
        mock_tf.keras.losses.Loss.assert_called_with(name=name, reduction="sum")

    def test_model_path_property(self):
        sim_loss = EncodingSimilarity(name="es", input_shape=(1,1,1)) # Dummy args for init
        # This is a concrete property in EncodingSimilarity
        self.assertEqual(sim_loss.model_path, ".tmp/20250115_095140/model.kid.h5")

    def test_call_method(self):
        sim_loss = EncodingSimilarity(name="es_call", input_shape=(32,32,3))
        
        y_true_mock = MagicMock(name="y_true_es_tensor")
        y_pred_mock = MagicMock(name="y_pred_es_tensor")
        
        # Mock the internal model that EncodingSimilarity.call uses
        mock_internal_keras_model = MagicMock(name="InternalEncodingModel")
        
        # This model is called twice: once with y_true, once with y_pred.
        # Mock the returned encodings.
        mock_y_true_encoding_tensor = MagicMock(name="y_true_encoding")
        mock_y_pred_encoding_tensor = MagicMock(name="y_pred_encoding")
        mock_internal_keras_model.side_effect = [mock_y_true_encoding_tensor, mock_y_pred_encoding_tensor]
        
        # Mock K.abs(y_true_encoding - y_pred_encoding)
        mock_abs_difference_tensor = MagicMock(name="AbsDifferenceTensor")
        mock_tf.keras.backend.abs.return_value = mock_abs_difference_tensor
        
        # Mock K.mean(...) of the absolute difference
        mock_mean_abs_diff_tensor = MagicMock(name="MeanAbsDifferenceTensor")
        mock_tf.keras.backend.mean.return_value = mock_mean_abs_diff_tensor

        # Patch the 'model' property (inherited from LossFromModel) to return our mock_internal_keras_model
        with patch.object(EncodingSimilarity, 'model', new_callable=PropertyMock, return_value=mock_internal_keras_model):
            # Call the loss function's call method
            result_tensor = sim_loss.call(y_true_mock, y_pred_mock)

            # Assertions
            # 1. The internal model was called with y_true and then with y_pred
            mock_internal_keras_model.assert_has_calls([
                call(y_true_mock),
                call(y_pred_mock)
            ])
            self.assertEqual(mock_internal_keras_model.call_count, 2)
            
            # 2. K.abs was called with (y_true_encoding - y_pred_encoding)
            # We need to ensure the subtraction happened. Let's assume tf operator overloading works.
            # The arguments to K.abs would be the result of (mock_y_true_encoding_tensor - mock_y_pred_encoding_tensor)
            mock_tf.keras.backend.abs.assert_called_once()
            # Example of checking the argument if the subtraction itself returned a mock:
            # mock_subtracted_tensor = MagicMock()
            # mock_y_true_encoding_tensor.__sub__ = MagicMock(return_value=mock_subtracted_tensor)
            # ... then check mock_tf.keras.backend.abs.assert_called_once_with(mock_subtracted_tensor)
            # For now, we trust the subtraction happens and K.abs gets some tensor.

            # 3. K.mean was called with the output of K.abs
            mock_tf.keras.backend.mean.assert_called_once_with(mock_abs_difference_tensor)
            
            # 4. The result of EncodingSimilarity.call is the result of K.mean
            self.assertIs(result_tensor, mock_mean_abs_diff_tensor)

    # Test for update_state, result, reset_state are inherited from LossFromModel
    # and tested in test_base_loss.py.

if __name__ == '__main__':
    unittest.main()
