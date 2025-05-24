import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

# Mock TensorFlow before it's imported by the modules under test
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.layers'] = mock_tf.keras.layers
sys.modules['tensorflow.keras.models'] = mock_tf.keras.models
sys.modules['tensorflow.keras.optimizers'] = mock_tf.keras.optimizers # For parent's compile

# Mock Rignak
sys.modules['Rignak.lazy_property'] = MagicMock()

# Mock custom modules (used by parent CategorizerWrapper)
mock_build_encoder_reg = MagicMock(name="BuildEncoderForRegression") # Use a distinct name if needed
sys.modules['src.modules.module'] = MagicMock(build_encoder=mock_build_encoder_reg)

# Mock losses (mae is used by RegressionWrapper, others by parent)
mock_loss_class_reg = MagicMock(name="LossClassForRegression")
mock_mae_func_reg = MagicMock(name="mae_func_for_regression")
# Parent CategorizerWrapper uses cross_entropy, one_minus_dice, std_difference
mock_cross_entropy_parent = MagicMock(name="cross_entropy_parent_mock")
mock_one_minus_dice_parent = MagicMock(name="one_minus_dice_parent_mock")
mock_std_diff_parent = MagicMock(name="std_diff_parent_mock")

sys.modules['src.losses.losses'] = MagicMock(
    Loss=mock_loss_class_reg,
    mae=mock_mae_func_reg,
    cross_entropy=mock_cross_entropy_parent, # For parent if its properties are accessed
    one_minus_dice=mock_one_minus_dice_parent,
    std_difference=mock_std_diff_parent
)

from models.image_to_tag.regression_wrapper import RegressionWrapper
# CategorizerWrapper is the parent class

class TestRegressionWrapper(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        mock_build_encoder_reg.reset_mock()
        mock_loss_class_reg.reset_mock()
        mock_mae_func_reg.reset_mock()
        mock_cross_entropy_parent.reset_mock() # Reset parent's mocks too

        # Keras layer/model mocks (as used by CategorizerWrapper's output_layer)
        self.MockInputLayer = MagicMock(name="InputLayerInstanceReg")
        self.MockEncoderOutputReg = MagicMock(name="EncoderOutputFromBuildEncoderReg")
        mock_build_encoder_reg.return_value = (self.MockEncoderOutputReg, []) # Parent expects tuple

        self.MockGlobalAvgPoolingLayerReg = MagicMock(name="GlobalAvgPoolingInstanceReg")
        self.MockDenseLayer1Reg = MagicMock(name="DenseLayer1InstanceReg")
        self.MockDenseLayerOutputReg = MagicMock(name="DenseLayerOutputInstanceReg")
        
        mock_tf.keras.layers.Input.return_value = self.MockInputLayer
        mock_tf.keras.layers.GlobalAveragePooling2D.return_value = self.MockGlobalAvgPoolingLayerReg
        mock_tf.keras.layers.Dense.side_effect = [self.MockDenseLayer1Reg, self.MockDenseLayerOutputReg]
        
        self.MockKerasModelReg = MagicMock(name="KerasModelInstanceReg")
        mock_tf.keras.Model.return_value = self.MockKerasModelReg

        # Mock training_generator and output_space (needed by parent CategorizerWrapper)
        self.mock_training_generator_reg = MagicMock(name="TrainingGeneratorReg")
        self.mock_output_space_reg = MagicMock(name="OutputSpaceReg")
        # For RegressionWrapper, output_shape is typically (1,) for a single regression value
        # or (N,) for N regression values.
        self.regression_output_dim = 1 
        self.wrapper_output_shape_reg = (self.regression_output_dim,)
        self.mock_output_space_reg.output_shape = self.wrapper_output_shape_reg
        # class_weights might be accessed by parent's loss if not careful, mock it
        self.mock_output_space_reg.class_weights = [1.0] # Dummy for parent
        self.mock_training_generator_reg.output_space = self.mock_output_space_reg
        
        self.input_shape_reg = (64, 64, 1) # Example input shape
        self.layer_kernels_reg = (16, 32) 
        self.n_stride_reg = 2

    def create_wrapper(self, **kwargs):
        params = {
            'input_shape': self.input_shape_reg,
            'layer_kernels': self.layer_kernels_reg,
            'n_stride': self.n_stride_reg,
            'training_generator': self.mock_training_generator_reg,
            **kwargs 
        }
        return RegressionWrapper(**params)

    def test_initialization_parameters(self):
        wrapper = self.create_wrapper()
        self.assertEqual(wrapper.input_shape, self.input_shape_reg)
        self.assertEqual(wrapper.layer_kernels, self.layer_kernels_reg)
        self.assertIs(wrapper.training_generator, self.mock_training_generator_reg)
        # output_shape is derived from training_generator.output_space by parent
        self.assertEqual(wrapper.output_shape, self.wrapper_output_shape_reg)
        
        self.assertNotIn('loss', wrapper.__dict__) # LazyProperty
        self.assertNotIn('metrics', wrapper.__dict__) # LazyProperty

    def test_loss_property_is_mae(self):
        wrapper = self.create_wrapper()
        loss_instance = wrapper.loss # Access property

        # RegressionWrapper's loss is Loss((losses.mae,))
        mock_loss_class_reg.assert_called_once_with((mock_mae_func_reg,))
        self.assertIsNotNone(loss_instance)
        # Ensure it's cached
        self.assertIs(wrapper.loss, loss_instance)

    def test_metrics_property_is_empty_tuple(self):
        wrapper = self.create_wrapper()
        metrics_tuple = wrapper.metrics # Access property

        self.assertEqual(metrics_tuple, ())
        # Ensure it's cached
        self.assertIs(wrapper.metrics, metrics_tuple)

    def test_output_layer_is_inherited_from_categorizerwrapper(self):
        # The output_layer structure (encoder -> GlobalAvgPool -> Dense -> Dense)
        # is defined in CategorizerWrapper. RegressionWrapper inherits this.
        # We test that it's built correctly, using RegressionWrapper's output_shape.
        wrapper = self.create_wrapper()
        
        # Access output_layer to trigger its construction
        output_layer_instance = wrapper.output_layer

        # 1. build_encoder call (from parent)
        mock_build_encoder_reg.assert_called_once_with(
            wrapper.input_layer,
            self.layer_kernels_reg,
            self.n_stride_reg
        )
        
        # 2. GlobalAveragePooling2D call (from parent)
        mock_tf.keras.layers.GlobalAveragePooling2D.assert_called_once()
        self.MockGlobalAvgPoolingLayerReg.assert_called_with(self.MockEncoderOutputReg)

        # 3. Dense layer calculations (from parent, using RegressionWrapper's output_shape)
        # n_intermediate = abs(int((layer_kernels[-1] - output_shape[-1]) / 2))
        # layer_kernels[-1] is 32, output_shape[-1] (regression_output_dim) is 1
        # n_intermediate = abs(int((32 - 1) / 2)) = abs(int(31 / 2)) = abs(int(15.5)) = 15
        expected_n_intermediate = 15
        
        mock_tf.keras.layers.Dense.assert_any_call(expected_n_intermediate, activation="relu")
        self.MockDenseLayer1Reg.assert_called_with(self.MockGlobalAvgPoolingLayerReg.return_value)
        
        # Output Dense layer uses regression_output_dim and "sigmoid" (from parent)
        # While "linear" might be more common for regression, the parent uses "sigmoid".
        # This test verifies it inherits that.
        mock_tf.keras.layers.Dense.assert_any_call(self.regression_output_dim, activation="sigmoid")
        self.MockDenseLayerOutputReg.assert_called_with(self.MockDenseLayer1Reg.return_value)
        
        self.assertIs(output_layer_instance, self.MockDenseLayerOutputReg)
        self.assertIs(wrapper.output_layer, output_layer_instance) # Cached

    def test_build_model_name_is_regression(self):
        # ModelWrapper sets the model name based on the class name.
        wrapper = self.create_wrapper()
        wrapper._input_layer = self.MockInputLayer # Manually set for build_model
        wrapper._output_layer = self.MockDenseLayerOutputReg # Manually set

        model = wrapper.build_model()

        mock_tf.keras.Model.assert_called_once_with(
            inputs=self.MockInputLayer,
            outputs=self.MockDenseLayerOutputReg,
            name="Regression" # Default name for RegressionWrapper
        )
        self.assertIs(model, self.MockKerasModelReg)

    def test_compile_model_uses_regression_loss_and_metrics(self):
        wrapper = self.create_wrapper()
        wrapper.model = self.MockKerasModelReg # Assign mock model
        
        mock_optimizer_instance = MagicMock(name="OptimizerInstanceReg")
        mock_tf.keras.optimizers.Adam.return_value = mock_optimizer_instance

        wrapper.compile(learning_rate=0.01) # Method from ModelWrapper (grandparent)

        mock_tf.keras.optimizers.Adam.assert_called_once_with(learning_rate=0.01)
        
        # Loss and metrics should be from RegressionWrapper's properties
        expected_loss_instance = wrapper.loss # Loss((losses.mae,))
        expected_metrics_tuple = wrapper.metrics # () - empty tuple
        
        self.MockKerasModelReg.compile.assert_called_once_with(
            optimizer=mock_optimizer_instance,
            loss=expected_loss_instance,
            metrics=expected_metrics_tuple
        )

if __name__ == '__main__':
    unittest.main()
