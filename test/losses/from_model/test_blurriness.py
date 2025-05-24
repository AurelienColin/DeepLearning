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
sys.modules['tensorflow.keras.backend'] = mock_tf.keras.backend # For K.mean

# Mock Rignak (used by parent LossFromModel)
sys.modules['Rignak.lazy_property'] = MagicMock()

# Mock custom_objects (used by parent LossFromModel when loading model)
sys.modules['src.modules.custom_objects'] = MagicMock(CUSTOM_OBJECTS={})


from losses.from_model.blurriness import Blurriness
# LossFromModel is the parent class, its testing is separate.

class TestBlurriness(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()

    def test_initialization(self):
        # Blurriness inherits __init__ from LossFromModel.
        # LossFromModel.__init__ takes name and input_shape.
        name = "test_blurriness_loss"
        input_shape = (128, 128, 3) # Example input shape
        
        # We pass reduction to tf.keras.losses.Loss parent
        blur_loss = Blurriness(name=name, input_shape=input_shape, reduction="none")

        self.assertEqual(blur_loss.name, name)
        self.assertEqual(blur_loss.input_shape, input_shape) # Set by LossFromModel's __init__
        # Check that parent tf.keras.losses.Loss.__init__ was called by LossFromModel's __init__
        mock_tf.keras.losses.Loss.assert_called_with(name=name, reduction="none")


    def test_model_path_property(self):
        blur_loss = Blurriness(name="b", input_shape=(1,1,1)) # Dummy args for init
        # This is a concrete property in Blurriness
        self.assertEqual(blur_loss.model_path, ".tmp/20250115_095140/model.blurry.h5")

    def test_call_method(self):
        blur_loss = Blurriness(name="b_call", input_shape=(32,32,3))
        
        y_true_mock = MagicMock(name="y_true_blur_tensor") # Not used by Blurriness.call
        y_pred_mock = MagicMock(name="y_pred_blur_tensor")
        
        # Mock the internal model that Blurriness.call uses
        mock_internal_keras_model = MagicMock(name="InternalBlurModel")
        # This internal model, when called, returns the blurriness value(s) for y_pred
        mock_predicted_blurriness_tensor = MagicMock(name="PredictedBlurrinessTensor")
        mock_internal_keras_model.return_value = mock_predicted_blurriness_tensor
        
        # Mock K.mean which is called on the result of model(y_pred)
        mock_mean_value_tensor = MagicMock(name="MeanBlurrinessValue")
        mock_tf.keras.backend.mean.return_value = mock_mean_value_tensor

        # Patch the 'model' property (inherited from LossFromModel) to return our mock_internal_keras_model
        with patch.object(Blurriness, 'model', new_callable=PropertyMock, return_value=mock_internal_keras_model):
            # Call the loss function's call method
            result_tensor = blur_loss.call(y_true_mock, y_pred_mock)

            # Assertions
            # 1. The internal model was called with y_pred
            mock_internal_keras_model.assert_called_once_with(y_pred_mock)
            
            # 2. K.mean was called with the output of the internal model
            mock_tf.keras.backend.mean.assert_called_once_with(mock_predicted_blurriness_tensor)
            
            # 3. The result of Blurriness.call is the result of K.mean
            self.assertIs(result_tensor, mock_mean_value_tensor)

    # Test for update_state, result, reset_state are inherited from LossFromModel
    # and tested in test_base_loss.py. We assume they work correctly if 'call' and 'tracker' work.

if __name__ == '__main__':
    unittest.main()
