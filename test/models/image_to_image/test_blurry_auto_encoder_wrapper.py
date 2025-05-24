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
sys.modules['tensorflow.keras.backend'] = mock_tf.keras.backend
sys.modules['tensorflow.keras.layers'] = mock_tf.keras.layers
sys.modules['tensorflow.keras.models'] = mock_tf.keras.models
sys.modules['tensorflow.keras.optimizers'] = mock_tf.keras.optimizers
sys.modules['tensorflow.keras.losses'] = mock_tf.keras.losses
sys.modules['tensorflow.keras.metrics'] = mock_tf.keras.metrics
sys.modules['tensorflow.keras.utils'] = mock_tf.keras.utils

# Mock AutoEncoderWrapper's dependencies that BlurryAutoEncoderWrapper might indirectly use
mock_build_encoder = MagicMock(name="BuildEncoder")
mock_build_decoder = MagicMock(name="BuildDecoder")
sys.modules['src.modules.module'] = MagicMock(build_encoder=mock_build_encoder, build_decoder=mock_build_decoder)

mock_edge_loss_func = MagicMock(name='edge_loss_func')
mock_mae_func = MagicMock(name='mae_func')
mock_fourth_channel_mae_func = MagicMock(name='fourth_channel_mae_func')

# Ensure Loss is properly mocked if needed, or use the real one if it's simple
from losses.losses import Loss
sys.modules['src.losses.losses'] = MagicMock(
    edge_loss=mock_edge_loss_func, 
    mae=mock_mae_func, 
    fourth_channel_mae=mock_fourth_channel_mae_func,
    Loss=Loss # Use the actual Loss class for combined losses
)

from models.image_to_image.blurry_auto_encoder_wrapper import BlurryAutoEncoderWrapper
# AutoEncoderWrapper is the parent, so its methods will be called.
# We will rely on its own tests for most of its behavior, and here focus on overrides.

class TestBlurryAutoEncoderWrapper(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        mock_build_encoder.reset_mock()
        mock_build_decoder.reset_mock()
        mock_edge_loss_func.reset_mock()
        mock_mae_func.reset_mock()
        mock_fourth_channel_mae_func.reset_mock()

        # Standard Keras layer mocks (similar to AutoEncoderWrapper test)
        self.MockInputLayer = MagicMock(name="InputLayerInstance")
        self.MockLambdaLayer = MagicMock(name="LambdaLayerInstance") # For encoded_layer
        self.MockConv2DLayer = MagicMock(name="Conv2DLayerInstance") # For output_layer
        
        mock_tf.keras.layers.Input.return_value = self.MockInputLayer
        mock_tf.keras.layers.Lambda.return_value = self.MockLambdaLayer
        mock_tf.keras.layers.Conv2D.return_value = self.MockConv2DLayer
        
        self.MockKerasModel = MagicMock(name="KerasModelInstance")
        mock_tf.keras.Model.return_value = self.MockKerasModel
        
        # Mock encoder/decoder build outputs (as parent AutoEncoderWrapper would use them)
        self.mock_encoder_output_layer = MagicMock(name="EncoderOutput")
        self.mock_encoder_inherited_layers = [MagicMock(name="EncInherited")]
        mock_build_encoder.return_value = (self.mock_encoder_output_layer, self.mock_encoder_inherited_layers)
        mock_build_decoder.return_value = MagicMock(name="DecoderOutputLayer") # Fed to Conv2D

        self.input_shape_rgb = (128, 128, 3)
        # BlurryAutoEncoderWrapper dynamically calculates its output_shape
        self.expected_output_shape_blurry = (128, 128, 3 + 1) # RGB + 1 sigma channel

        self.layer_kernels = (32, 64, 128)
        self.n_stride = 2

    def create_wrapper(self, input_shape, **kwargs):
        # Note: BlurryAutoEncoderWrapper determines its own output_shape
        params = {
            'input_shape': input_shape,
            'layer_kernels': self.layer_kernels,
            'n_stride': self.n_stride,
            **kwargs
        }
        # It does not take 'output_shape' as a direct init param for its core logic,
        # but AutoEncoderWrapper's __init__ might, so we pass it if provided in kwargs.
        # However, BlurryAEW's @LazyProperty for output_shape will override it.
        if 'output_shape' not in params and '_output_shape' not in params:
             # Pass the input_shape as a base, AutoEncoderWrapper's __init__ might use it
             # before Blurry's LazyProperty for output_shape takes over.
             # Or, more accurately, pass what AEW would expect for its _output_shape if not blurry.
             params['_output_shape'] = input_shape


        return BlurryAutoEncoderWrapper(**params)

    def test_initialization_parameters(self):
        # We are mostly testing that parameters are passed to AutoEncoderWrapper's init
        # and that BlurryAutoEncoderWrapper specific properties are not yet set.
        wrapper = self.create_wrapper(self.input_shape_rgb)
        self.assertEqual(wrapper.input_shape, self.input_shape_rgb)
        # _output_shape is internal to AutoEncoderWrapper, might be set by its __init__
        # The public output_shape is a LazyProperty in BlurryAEW
        self.assertIsNone(wrapper.__dict__.get('output_shape', None)) # Check if lazy prop cached
        self.assertEqual(wrapper.layer_kernels, self.layer_kernels)
        self.assertEqual(wrapper.n_stride, self.n_stride)

    def test_output_shape_property(self):
        wrapper = self.create_wrapper(self.input_shape_rgb)
        # Access the property
        oshape = wrapper.output_shape
        self.assertEqual(oshape, self.expected_output_shape_blurry)
        # Should be cached
        self.assertIs(wrapper.output_shape, oshape)
        self.assertIn('output_shape', wrapper.__dict__) # LazyProperty cached

    def test_metrics_property(self):
        wrapper_rgb_input = self.create_wrapper(self.input_shape_rgb)
        
        # Parent (AutoEncoderWrapper) metrics for RGB (edge_loss, mae)
        expected_parent_metrics_rgb = (mock_edge_loss_func, mock_mae_func)
        
        # BlurryAutoEncoderWrapper adds fourth_channel_mae
        expected_metrics = (*expected_parent_metrics_rgb, mock_fourth_channel_mae_func)
        
        metrics = wrapper_rgb_input.metrics
        self.assertEqual(metrics, expected_metrics)
        # Should be cached
        self.assertIs(wrapper_rgb_input.metrics, metrics)
    
    def test_metrics_property_mono_input(self):
        # If input is monochrome, parent AEW returns (mae,)
        input_shape_mono = (64, 64, 1)
        wrapper_mono_input = self.create_wrapper(input_shape_mono)
        
        expected_parent_metrics_mono = (mock_mae_func,)
        expected_metrics_blurry_mono = (*expected_parent_metrics_mono, mock_fourth_channel_mae_func)
        
        metrics = wrapper_mono_input.metrics
        self.assertEqual(metrics, expected_metrics_blurry_mono)

    def test_loss_is_inherited_from_autoencoderwrapper(self):
        # BlurryAEW does not override 'loss', so it uses AEW's loss logic.
        # AEW's loss logic depends on its own output_shape.
        # BlurryAEW's output_shape is (H, W, C+1).
        # If input C=3, Blurry output C=4. AEW loss for 4 channels is (edge_loss, mae)
        wrapper_rgb_input = self.create_wrapper(self.input_shape_rgb) # input C=3
        
        # wrapper_rgb_input.output_shape will be (H,W,4)
        self.assertEqual(wrapper_rgb_input.output_shape[-1], 4) 
        
        loss_func = wrapper_rgb_input.loss # Uses AEW's loss logic with new output_shape
        
        self.assertIsInstance(loss_func, Loss)
        self.assertEqual(loss_func.losses, (mock_edge_loss_func, mock_mae_func))
        self.assertEqual(loss_func.loss_weights, (0.1, 1))

        # If input C=1, Blurry output C=2. AEW loss for 2 channels is mae.
        input_shape_mono = (64, 64, 1)
        wrapper_mono_input = self.create_wrapper(input_shape_mono) # input C=1
        self.assertEqual(wrapper_mono_input.output_shape[-1], 2)

        loss_func_mono = wrapper_mono_input.loss
        self.assertIs(loss_func_mono, mock_mae_func)


    def test_build_model_uses_blurry_output_shape(self):
        # This test ensures that when the model is built, the final Conv2D layer
        # in AutoEncoderWrapper's output_layer uses the channel count from
        # BlurryAutoEncoderWrapper's overridden output_shape.
        wrapper = self.create_wrapper(self.input_shape_rgb)
        
        # Access output_layer to trigger its construction.
        # This happens inside AutoEncoderWrapper.
        _ = wrapper.output_layer 

        # The last Conv2D in AutoEncoderWrapper's output_layer should use
        # self.output_shape[-1] for its channel count.
        # For BlurryAEW, this output_shape is (H,W,C+1).
        mock_tf.keras.layers.Conv2D.assert_called_with(
            self.expected_output_shape_blurry[-1], # Channels = input_channels + 1
            activation="sigmoid",
            kernel_size=1
        )
        
        # Now, actually build the model
        wrapper.build_model()
        mock_tf.keras.Model.assert_called_once_with(
            inputs=self.MockInputLayer, # From AutoEncoderWrapper's input_layer property
            outputs=self.MockConv2DLayer, # From AutoEncoderWrapper's output_layer property
            name="BlurryAutoEncoder" # Name is overridden in BlurryAEW's @dataclass defaults
        )
        self.assertIs(wrapper.model, self.MockKerasModel)

    def test_compile_model_uses_blurry_metrics_and_loss(self):
        wrapper = self.create_wrapper(self.input_shape_rgb)
        wrapper.model = self.MockKerasModel # Assign a mock model instance
        
        mock_optimizer_instance = MagicMock()
        mock_tf.keras.optimizers.Adam.return_value = mock_optimizer_instance

        wrapper.compile(learning_rate=0.002)

        mock_tf.keras.optimizers.Adam.assert_called_once_with(learning_rate=0.002)
        
        # Expected loss for RGB input (output C=4) is combined (edge_loss, mae)
        expected_loss_obj = Loss((mock_edge_loss_func, mock_mae_func), loss_weights=(0.1,1))
        # Expected metrics for RGB input: (edge_loss, mae, fourth_channel_mae)
        expected_metrics_obj = (mock_edge_loss_func, mock_mae_func, mock_fourth_channel_mae_func)
        
        args, kwargs = self.MockKerasModel.compile.call_args
        
        compiled_loss = kwargs['loss']
        self.assertIsInstance(compiled_loss, Loss)
        self.assertEqual(compiled_loss.losses, expected_loss_obj.losses)
        self.assertEqual(compiled_loss.loss_weights, expected_loss_obj.loss_weights)
        
        self.assertEqual(kwargs['optimizer'], mock_optimizer_instance)
        self.assertEqual(kwargs['metrics'], expected_metrics_obj)


if __name__ == '__main__':
    unittest.main()
