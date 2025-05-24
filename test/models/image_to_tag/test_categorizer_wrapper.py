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

# Mock custom modules
mock_build_encoder = MagicMock(name="BuildEncoder")
sys.modules['src.modules.module'] = MagicMock(build_encoder=mock_build_encoder)

# Mock losses
mock_loss_class = MagicMock(name="LossClass")
mock_cross_entropy_func = MagicMock(name="cross_entropy_func")
mock_one_minus_dice_func = MagicMock(name="one_minus_dice_func")
mock_std_difference_func = MagicMock(name="std_difference_func")
sys.modules['src.losses.losses'] = MagicMock(
    Loss=mock_loss_class,
    cross_entropy=mock_cross_entropy_func,
    one_minus_dice=mock_one_minus_dice_func,
    std_difference=mock_std_difference_func
)

from models.image_to_tag.categorizer_wrapper import CategorizerWrapper
# ModelWrapper is the parent class

class TestCategorizerWrapper(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        mock_build_encoder.reset_mock()
        mock_loss_class.reset_mock()
        mock_cross_entropy_func.reset_mock()
        mock_one_minus_dice_func.reset_mock()
        mock_std_difference_func.reset_mock()

        # Keras layer/model mocks
        self.MockInputLayer = MagicMock(name="InputLayerInstance")
        self.MockEncoderOutput = MagicMock(name="EncoderOutputFromBuildEncoder")
        self.MockEncoderInherited = [MagicMock(name="EncoderInheritedLayer")]
        mock_build_encoder.return_value = (self.MockEncoderOutput, self.MockEncoderInherited)

        self.MockGlobalAvgPoolingLayer = MagicMock(name="GlobalAvgPoolingInstance")
        self.MockDenseLayer1 = MagicMock(name="DenseLayer1Instance")
        self.MockDenseLayerOutput = MagicMock(name="DenseLayerOutputInstance")
        
        mock_tf.keras.layers.Input.return_value = self.MockInputLayer
        mock_tf.keras.layers.GlobalAveragePooling2D.return_value = self.MockGlobalAvgPoolingLayer
        # Dense calls are sequential
        mock_tf.keras.layers.Dense.side_effect = [self.MockDenseLayer1, self.MockDenseLayerOutput]
        
        self.MockKerasModel = MagicMock(name="KerasModelInstance")
        mock_tf.keras.Model.return_value = self.MockKerasModel

        # Mock training_generator and output_space
        self.mock_training_generator = MagicMock(name="TrainingGenerator")
        self.mock_output_space = MagicMock(name="OutputSpace")
        self.mock_output_space.class_weights = [0.1, 0.9] # Example class weights
        self.mock_training_generator.output_space = self.mock_output_space
        
        self.input_shape = (128, 128, 3)
        # output_shape for CategorizerWrapper comes from training_generator.output_space.output_shape
        self.num_classes = 5 
        self.wrapper_output_shape = (self.num_classes,) # (num_classes,)
        self.mock_output_space.output_shape = self.wrapper_output_shape

        self.layer_kernels = (32, 64, 128) # Default, but specify for clarity
        self.n_stride = 2 # Default from ModelWrapper

    def create_wrapper(self, **kwargs):
        params = {
            'input_shape': self.input_shape,
            # output_shape is implicitly set via training_generator.output_space
            'layer_kernels': self.layer_kernels,
            'n_stride': self.n_stride,
            'training_generator': self.mock_training_generator,
            **kwargs 
        }
        # CategorizerWrapper's __init__ passes _output_shape to ModelWrapper
        # ModelWrapper then uses this to set its public output_shape if training_generator is None
        # But CategorizerWrapper ensures training_generator is passed, and ModelWrapper then uses
        # training_generator.output_space.output_shape.
        # So, the output_shape is effectively determined by mock_output_space.output_shape
        return CategorizerWrapper(**params)

    def test_initialization_parameters(self):
        wrapper = self.create_wrapper()
        self.assertEqual(wrapper.input_shape, self.input_shape)
        self.assertEqual(wrapper.layer_kernels, self.layer_kernels)
        self.assertEqual(wrapper.n_stride, self.n_stride)
        self.assertIs(wrapper.training_generator, self.mock_training_generator)
        # output_shape should be derived from training_generator.output_space
        self.assertEqual(wrapper.output_shape, self.wrapper_output_shape)
        
        # Check that LazyProperties are not yet in __dict__
        self.assertNotIn('loss', wrapper.__dict__)
        self.assertNotIn('metrics', wrapper.__dict__)
        self.assertNotIn('output_layer', wrapper.__dict__)

    def test_loss_property(self):
        wrapper = self.create_wrapper()
        loss_instance = wrapper.loss # Access property

        mock_loss_class.assert_called_once_with(
            (mock_cross_entropy_func,), 
            class_weights=self.mock_output_space.class_weights
        )
        self.assertIsNotNone(loss_instance)
        self.assertIs(wrapper.loss, loss_instance) # Cached

    def test_metrics_property(self):
        wrapper = self.create_wrapper()
        metrics_tuple = wrapper.metrics # Access property

        self.assertEqual(metrics_tuple, (mock_one_minus_dice_func, mock_std_difference_func))
        self.assertIs(wrapper.metrics, metrics_tuple) # Cached

    def test_output_layer_property_structure(self):
        wrapper = self.create_wrapper()
        
        # Access output_layer to trigger its construction
        output_layer_instance = wrapper.output_layer

        # 1. build_encoder call
        mock_build_encoder.assert_called_once_with(
            wrapper.input_layer, # From ModelWrapper's __post_init__
            self.layer_kernels,
            self.n_stride
        )
        
        # 2. GlobalAveragePooling2D call
        mock_tf.keras.layers.GlobalAveragePooling2D.assert_called_once()
        # Check it was called on the output of build_encoder
        self.MockGlobalAvgPoolingLayer.assert_called_with(self.MockEncoderOutput)

        # 3. Dense layer calculations
        # n_intermediate = abs(int((self.layer_kernels[-1] - self.output_shape[-1]) / 2))
        # layer_kernels[-1] is 128, output_shape[-1] (num_classes) is 5
        # n_intermediate = abs(int((128 - 5) / 2)) = abs(int(123 / 2)) = abs(int(61.5)) = 61
        expected_n_intermediate = 61
        
        # First Dense layer call
        mock_tf.keras.layers.Dense.assert_any_call(expected_n_intermediate, activation="relu")
        # Check it was called on the output of GlobalAveragePooling2D
        self.MockDenseLayer1.assert_called_with(self.MockGlobalAvgPoolingLayer.return_value)
        
        # Second Dense layer (output layer) call
        mock_tf.keras.layers.Dense.assert_any_call(self.num_classes, activation="sigmoid")
        # Check it was called on the output of the first Dense layer
        self.MockDenseLayerOutput.assert_called_with(self.MockDenseLayer1.return_value)
        
        self.assertIs(output_layer_instance, self.MockDenseLayerOutput)
        self.assertIs(wrapper.output_layer, output_layer_instance) # Cached
        # Ensure build_encoder and layer constructors are only called once
        mock_build_encoder.assert_called_once()
        mock_tf.keras.layers.GlobalAveragePooling2D.assert_called_once()
        self.assertEqual(mock_tf.keras.layers.Dense.call_count, 2)


    def test_build_model(self):
        wrapper = self.create_wrapper()
        
        # Manually set _input_layer and _output_layer as they are accessed by build_model
        # In a real scenario, accessing wrapper.output_layer would set wrapper._output_layer.
        wrapper._input_layer = self.MockInputLayer 
        # To correctly mock this, we need to ensure that wrapper.output_layer (the property)
        # has been called before build_model, or manually set _output_layer
        wrapper._output_layer = self.MockDenseLayerOutput # Result of output_layer property
        
        model = wrapper.build_model()

        mock_tf.keras.Model.assert_called_once_with(
            inputs=self.MockInputLayer,
            outputs=self.MockDenseLayerOutput,
            name="Categorizer" # Default name from ModelWrapper based on class name
        )
        self.assertIs(model, self.MockKerasModel)
        self.assertIs(wrapper.model, self.MockKerasModel) # Assigned to property

    def test_compile_model_uses_wrapper_loss_and_metrics(self):
        wrapper = self.create_wrapper()
        wrapper.model = self.MockKerasModel # Assign mock model
        
        # Mock optimizer
        mock_optimizer_instance = MagicMock(name="OptimizerInstance")
        mock_tf.keras.optimizers.Adam.return_value = mock_optimizer_instance

        wrapper.compile(learning_rate=0.001) # Method from ModelWrapper

        mock_tf.keras.optimizers.Adam.assert_called_once_with(learning_rate=0.001)
        
        # The loss and metrics should be the instances from CategorizerWrapper's properties
        expected_loss_instance = wrapper.loss 
        expected_metrics_tuple = wrapper.metrics
        
        self.MockKerasModel.compile.assert_called_once_with(
            optimizer=mock_optimizer_instance,
            loss=expected_loss_instance,
            metrics=expected_metrics_tuple
        )

if __name__ == '__main__':
    unittest.main()
