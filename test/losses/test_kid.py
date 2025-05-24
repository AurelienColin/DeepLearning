import unittest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock TensorFlow before it's imported by the modules under test
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.backend'] = mock_tf.keras.backend
sys.modules['tensorflow.keras.layers'] = mock_tf.keras.layers
sys.modules['tensorflow.keras.models'] = mock_tf.keras.models
sys.modules['tensorflow.keras.metrics'] = mock_tf.keras.metrics
sys.modules['tensorflow.keras.applications'] = mock_tf.keras.applications
sys.modules['tensorflow.keras.applications.inception_v3'] = mock_tf.keras.applications.inception_v3
# Mock experimental preprocessing
mock_tf.keras.layers.experimental = MagicMock()
mock_tf.keras.layers.experimental.preprocessing = MagicMock()


from losses.kid import KID

class TestKernelInceptionDistance(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        # Reset specific application mocks if they are stateful
        mock_tf.keras.applications.InceptionV3.reset_mock()
        mock_tf.keras.layers.experimental.preprocessing.Resizing.reset_mock()


    def test_initialization_with_default_encoder(self):
        input_shape = (299, 299, 3) # Shape expected by InceptionV3 sometimes
        kid_metric = KID(name="kid_test", input_shape=input_shape)

        self.assertEqual(kid_metric.name, "kid_test")
        mock_tf.keras.metrics.Mean.assert_called_once_with(name="kid_tracker")
        self.assertIsNotNone(kid_metric.kid_tracker)
        self.assertIsNotNone(kid_metric.encoder)
        
        # Check that the default encoder was attempted to be built
        # (signified by layers being created)
        mock_tf.keras.layers.Input.assert_called_once_with(shape=input_shape)
        mock_tf.keras.layers.experimental.preprocessing.Resizing.assert_called_once_with(
            height=KID.KID_IMAGE_SIZE, width=KID.KID_IMAGE_SIZE
        )
        mock_tf.keras.applications.InceptionV3.assert_called_once_with(
            include_top=False,
            input_shape=(KID.KID_IMAGE_SIZE, KID.KID_IMAGE_SIZE, 3),
            weights="imagenet",
        )
        mock_tf.keras.layers.GlobalAveragePooling2D.assert_called_once()
        mock_tf.keras.models.Sequential.assert_called() # Encoder is a Sequential model


    def test_initialization_with_custom_layers(self):
        input_shape = (32, 32, 3)
        mock_custom_input_layer = MagicMock(name="CustomInput")
        mock_custom_conv_layer = MagicMock(name="CustomConv")
        custom_layers = [mock_custom_conv_layer] # The input layer is added by KID

        mock_tf.keras.layers.Input.return_value = mock_custom_input_layer
        
        kid_metric = KID(name="kid_custom", input_shape=input_shape, layers=custom_layers)

        mock_tf.keras.layers.Input.assert_called_once_with(shape=input_shape)
        # Default encoder should not be called
        mock_tf.keras.applications.InceptionV3.assert_not_called()
        
        # Sequential model should be called with the custom layers prepended by the input layer
        mock_tf.keras.models.Sequential.assert_called_once_with((mock_custom_input_layer, mock_custom_conv_layer))
        self.assertIsNotNone(kid_metric.encoder)


    def test_polynomial_kernel(self):
        kid_metric = KID(name="kid_poly", input_shape=(1,1,1)) # Dummy shape for init

        mock_features_1 = MagicMock(name="features1_tensor")
        mock_features_2 = MagicMock(name="features2_tensor")
        
        # Mock K.shape to return a mock tensor for feature_dimensions
        mock_shape_tensor = MagicMock(name="shape_tensor")
        mock_shape_tensor.__getitem__.return_value = MagicMock(name="feature_dim_val") # for K.shape()[1]
        mock_tf.keras.backend.shape.return_value = mock_shape_tensor
        
        mock_feature_dim_casted = MagicMock(name="feature_dim_casted")
        mock_tf.keras.backend.cast.return_value = mock_feature_dim_casted
        
        mock_transpose_features_2 = MagicMock(name="transpose_f2")
        mock_tf.keras.backend.transpose.return_value = mock_transpose_features_2
        
        # Mock matrix multiplication (features_1 @ transpose_features_2)
        mock_matmul_result = MagicMock(name="matmul_result")
        mock_features_1.__matmul__ = MagicMock(return_value=mock_matmul_result)

        # Mock the rest of the calculation
        # (matmul_result / feature_dimensions + 1.0) ** 3.0
        # For simplicity, we don't mock each individual tf math op here, but assume they work.
        # The key is that the inputs are used correctly.
        # If specific intermediate values were needed, we'd mock division, addition, power.
        expected_kernel_result_tensor = MagicMock(name="PolyKernelOutput")
        
        # Let's assume the sequence of ops results in the final tensor.
        # For a more rigorous test, one might mock the division, addition, and power steps.
        # Here, we'll just ensure the inputs are processed as expected.
        # Example: (X / Y + 1)**3 -> if X, Y are mocks, X.__truediv__(Y).__add__(1.0).__pow__(3.0)
        # This can get very verbose. We will assume the math ops are correct if inputs are correct.
        
        # Let's mock the final operation in the chain for simplicity of assertion
        mock_intermediate_add = MagicMock(name="IntermediateAdd")
        mock_intermediate_div = MagicMock(name="IntermediateDiv")
        mock_matmul_result.__truediv__ = MagicMock(return_value=mock_intermediate_div)
        mock_intermediate_div.__add__ = MagicMock(return_value=mock_intermediate_add)
        mock_intermediate_add.__pow__ = MagicMock(return_value=expected_kernel_result_tensor)


        result_tensor = kid_metric.polynomial_kernel(mock_features_1, mock_features_2)

        mock_tf.keras.backend.shape.assert_called_once_with(mock_features_1)
        mock_tf.keras.backend.cast.assert_called_once_with(mock_shape_tensor.__getitem__.return_value, dtype="float32")
        mock_tf.keras.backend.transpose.assert_called_once_with(mock_features_2)
        mock_features_1.__matmul__.assert_called_once_with(mock_transpose_features_2)
        
        # Check that the sequence of operations was called
        mock_matmul_result.__truediv__.assert_called_once_with(mock_feature_dim_casted)
        mock_intermediate_div.__add__.assert_called_once_with(1.0)
        mock_intermediate_add.__pow__.assert_called_once_with(3.0)

        self.assertIs(result_tensor, expected_kernel_result_tensor)


    def test_update_state(self):
        input_shape = (32, 32, 3)
        kid_metric = KID(name="kid_update", input_shape=input_shape)

        # Mock encoder and its return values
        mock_encoder_model = MagicMock(name="EncoderModelInstance")
        kid_metric.encoder = mock_encoder_model # Replace the actual encoder with a mock
        
        mock_real_features = MagicMock(name="RealFeatures")
        mock_generated_features = MagicMock(name="GeneratedFeatures")
        mock_encoder_model.side_effect = [mock_real_features, mock_generated_features]

        # Mock real and generated images
        mock_real_images = MagicMock(name="RealImages")
        mock_generated_images = MagicMock(name="GeneratedImages")
        
        # Mock polynomial_kernel results
        mock_kernel_real_val = MagicMock(name="KernelReal")
        mock_kernel_gen_val = MagicMock(name="KernelGen")
        mock_kernel_cross_val = MagicMock(name="KernelCross")
        
        # Patch the instance method polynomial_kernel
        with patch.object(KID, 'polynomial_kernel', side_effect=[mock_kernel_real_val, mock_kernel_gen_val, mock_kernel_cross_val]) as mock_poly_kernel_method:
            # Mock batch_size related calculations
            # real_features.shape[0] -> mock this
            mock_batch_size_val = 5 
            mock_real_features.shape = [mock_batch_size_val, 10] # Example shape
            
            mock_batch_size_f_val = MagicMock(name="BatchSizeF")
            mock_tf.keras.backend.cast.return_value = mock_batch_size_f_val
            
            # Mock K.eye, K.sum, K.mean
            mock_eye_val = MagicMock(name="EyeMatrix")
            mock_tf.keras.backend.eye.return_value = mock_eye_val
            
            mock_sum_real_val = MagicMock(name="SumKernelReal")
            mock_sum_gen_val = MagicMock(name="SumKernelGen")
            mock_tf.keras.backend.sum.side_effect = [mock_sum_real_val, mock_sum_gen_val]
            
            mock_mean_cross_val = MagicMock(name="MeanKernelCross")
            mock_tf.keras.backend.mean.return_value = mock_mean_cross_val

            # Mock kid_tracker.update_state
            kid_metric.kid_tracker = MagicMock(name="KidTrackerInstance")

            # Mock arithmetic operations for KID calculation
            # For simplicity, we assume these operations will work if inputs are correct.
            # A more detailed test would mock each __add__, __sub__, __mul__, __truediv__.
            # Let's assume the final computed KID value is some mock object.
            mock_final_kid_value = MagicMock(name="FinalKIDValue")

            # To mock `mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross`
            # This gets complicated. Let's focus on inputs to kid_tracker.update_state.
            # We'll assume the arithmetic ops are called.
            
            # Call update_state
            kid_metric.update_state(mock_real_images, mock_generated_images)

            # Assert encoder calls
            mock_encoder_model.assert_any_call(mock_real_images, training=False)
            mock_encoder_model.assert_any_call(mock_generated_images, training=False)
            
            # Assert polynomial_kernel calls
            mock_poly_kernel_method.assert_any_call(mock_real_features, mock_real_features)
            mock_poly_kernel_method.assert_any_call(mock_generated_features, mock_generated_features)
            mock_poly_kernel_method.assert_any_call(mock_real_features, mock_generated_features)

            # Assert Keras backend calls for calculations
            mock_tf.keras.backend.cast.assert_called_with(mock_batch_size_val, dtype="float32")
            mock_tf.keras.backend.eye.assert_called_with(mock_batch_size_val)
            mock_tf.keras.backend.sum.assert_any_call(mock_kernel_real_val * (1.0 - mock_eye_val))
            mock_tf.keras.backend.sum.assert_any_call(mock_kernel_gen_val * (1.0 - mock_eye_val))
            mock_tf.keras.backend.mean.assert_called_once_with(mock_kernel_cross_val)
            
            # Assert kid_tracker update
            # The argument to update_state is the result of:
            # mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
            # This is hard to mock perfectly without running TF. We check that it's called.
            kid_metric.kid_tracker.update_state.assert_called_once() 
            # We could verify the argument if we mocked all intermediate arithmetic results.
            # For example, if:
            # mock_mean_real = mock_sum_real_val / (mock_batch_size_f_val * (mock_batch_size_f_val - 1.0))
            # ... and so on for mean_generated and mean_cross.
            # Then the arg would be mock_mean_real + mock_mean_generated - 2.0 * mock_mean_cross_val


    def test_result(self):
        kid_metric = KID(name="kid_res", input_shape=(1,1,1))
        kid_metric.kid_tracker = MagicMock(name="KidTrackerInstance")
        mock_tracker_result = 0.5
        kid_metric.kid_tracker.result.return_value = mock_tracker_result
        
        self.assertEqual(kid_metric.result(), mock_tracker_result)
        kid_metric.kid_tracker.result.assert_called_once()

    def test_reset_state(self):
        kid_metric = KID(name="kid_reset", input_shape=(1,1,1))
        kid_metric.kid_tracker = MagicMock(name="KidTrackerInstance")
        
        kid_metric.reset_state()
        kid_metric.kid_tracker.reset_state.assert_called_once()


if __name__ == '__main__':
    unittest.main()
