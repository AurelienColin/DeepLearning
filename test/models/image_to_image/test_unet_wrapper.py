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
# ... (add other tf submodules if UnetWrapper or its parent use them directly)

# Mock AutoEncoderWrapper's dependencies that UnetWrapper might indirectly use or override
mock_build_encoder = MagicMock(name="BuildEncoder")
mock_build_decoder = MagicMock(name="BuildDecoder")
sys.modules['src.modules.module'] = MagicMock(build_encoder=mock_build_encoder, build_decoder=mock_build_decoder)

mock_edge_loss_func = MagicMock(name='edge_loss_func')
mock_mae_func = MagicMock(name='mae_func')
from losses.losses import Loss # Assuming Loss is a simple class
sys.modules['src.losses.losses'] = MagicMock(
    edge_loss=mock_edge_loss_func, 
    mae=mock_mae_func,
    Loss=Loss
)

from models.image_to_image.unet_wrapper import UnetWrapper
# AutoEncoderWrapper is the parent. We assume its core functionalities are tested.
# We focus on what UnetWrapper changes or how its changes affect behavior.

class TestUnetWrapper(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        mock_build_encoder.reset_mock()
        mock_build_decoder.reset_mock()
        mock_edge_loss_func.reset_mock()
        mock_mae_func.reset_mock()
        
        # Keras layer/model mocks
        self.MockInputLayer = MagicMock(name="InputLayerInstance")
        self.MockLambdaLayerForEncoder = MagicMock(name="LambdaLayerForEncoder") # For encoded_layer
        self.MockConv2DLayerForOutput = MagicMock(name="Conv2DLayerForOutput") # For output_layer
        
        mock_tf.keras.layers.Input.return_value = self.MockInputLayer
        # AutoEncoderWrapper.set_encoded_layers uses a Lambda for tanh
        mock_tf.keras.layers.Lambda.return_value = self.MockLambdaLayerForEncoder
        # AutoEncoderWrapper.output_layer uses Conv2D
        mock_tf.keras.layers.Conv2D.return_value = self.MockConv2DLayerForOutput
        
        self.MockKerasModel = MagicMock(name="KerasModelInstance")
        mock_tf.keras.Model.return_value = self.MockKerasModel
        
        # Mock return values for encoder/decoder builders
        self.mock_encoder_output_actual = MagicMock(name="EncoderOutputActual") # Output from build_encoder before Lambda
        self.mock_encoder_inherited_layers_list = [MagicMock(name="EncInherited1"), MagicMock(name="EncInherited2")]
        mock_build_encoder.return_value = (self.mock_encoder_output_actual, self.mock_encoder_inherited_layers_list)

        self.mock_decoder_output_actual = MagicMock(name="DecoderOutputActual") # Output from build_decoder before Conv2D
        mock_build_decoder.return_value = self.mock_decoder_output_actual

        self.input_shape = (128, 128, 3)
        self.output_shape = (128, 128, 3) # Default, can be changed by specific tests
        self.layer_kernels = (32, 64)
        self.n_stride = 2

    def create_wrapper(self, **kwargs_autoencoder):
        params = {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape, # Or _output_shape for parent
            'layer_kernels': self.layer_kernels,
            'n_stride': self.n_stride,
            **kwargs_autoencoder
        }
        return UnetWrapper(**params)

    def test_initialization_parameters_passed_to_parent(self):
        # This just ensures UnetWrapper can be initialized and params are likely passed.
        # Parent AutoEncoderWrapper's tests cover its own init logic.
        wrapper = self.create_wrapper()
        self.assertEqual(wrapper.input_shape, self.input_shape)
        self.assertEqual(wrapper.output_shape, self.output_shape)
        self.assertEqual(wrapper.layer_kernels, self.layer_kernels)
        self.assertEqual(wrapper.n_stride, self.n_stride)
        # Lazy properties from parent or self are not yet set
        self.assertNotIn('encoded_inherited_layers', wrapper.__dict__)
        self.assertNotIn('encoded_layer', wrapper.__dict__) # From parent
        self.assertNotIn('output_layer', wrapper.__dict__) # From parent

    def test_encoded_inherited_layers_property_returns_actual_layers(self):
        wrapper = self.create_wrapper()
        
        # Accessing 'encoded_inherited_layers' will trigger 'set_encoded_layers'
        # (which is defined in AutoEncoderWrapper) via the property's internal call.
        inherited_layers = wrapper.encoded_inherited_layers
        
        # 1. Check that set_encoded_layers was called (which calls build_encoder)
        mock_build_encoder.assert_called_once_with(
            wrapper.input_layer, # Created in ModelWrapper's __post_init__
            self.layer_kernels,
            self.n_stride
        )
        # 2. Check that the returned value is what build_encoder provided
        self.assertIs(inherited_layers, self.mock_encoder_inherited_layers_list)
        
        # 3. Check that it's cached
        self.assertIs(wrapper.encoded_inherited_layers, inherited_layers)
        # set_encoded_layers should only be called once due to caching by LazyProperty
        mock_build_encoder.assert_called_once() 
        
        # Ensure the internal _encoded_inherited_layers (from AutoEncoderWrapper) was set
        self.assertIs(wrapper._encoded_inherited_layers, self.mock_encoder_inherited_layers_list)


    def test_output_layer_construction_uses_unet_inherited_layers(self):
        wrapper = self.create_wrapper()

        # Access encoded_layer first to set up the encoder part.
        # This also populates _encoded_layer and _encoded_inherited_layers.
        encoder_output_for_decoder = wrapper.encoded_layer # This is the Lambda(tanh) layer
        
        # Now, access UnetWrapper's 'encoded_inherited_layers' property.
        # This is what should be passed to build_decoder.
        unet_specific_inherited_layers = wrapper.encoded_inherited_layers
        self.assertIs(unet_specific_inherited_layers, self.mock_encoder_inherited_layers_list)

        # Access the output_layer property (defined in AutoEncoderWrapper)
        _ = wrapper.output_layer

        # Assert that build_decoder was called with the UnetWrapper's version of 
        # encoded_inherited_layers (i.e., the actual list, not an empty one).
        mock_build_decoder.assert_called_once_with(
            encoder_output_for_decoder,
            self.mock_encoder_inherited_layers_list, # <<< Key assertion for UnetWrapper
            self.layer_kernels,
            self.n_stride
        )
        
        # The rest of output_layer construction is as per AutoEncoderWrapper
        mock_tf.keras.layers.Conv2D.assert_called_once_with(
            self.output_shape[-1], # Number of output channels
            activation="sigmoid",
            kernel_size=1
        )
        self.MockConv2DLayerForOutput.assert_called_with(self.mock_decoder_output_actual)

    def test_build_model_name_is_unet(self):
        # ModelWrapper sets the model name based on the class name if not provided.
        wrapper = self.create_wrapper()
        # Set necessary components for model building
        wrapper._input_layer = self.MockInputLayer
        wrapper._output_layer = self.MockConv2DLayerForOutput

        model = wrapper.build_model()

        mock_tf.keras.Model.assert_called_once_with(
            inputs=self.MockInputLayer,
            outputs=self.MockConv2DLayerForOutput,
            name="Unet" # Default name for UnetWrapper
        )
        self.assertIs(model, self.MockKerasModel)

    # Other functionalities like loss, metrics, compile, etc., are inherited directly
    # from AutoEncoderWrapper and should be covered by its tests, unless UnetWrapper
    # overrides them or its changes (like encoded_inherited_layers) indirectly affect them.
    # For instance, if loss/metrics calculation depended on the exact structure of
    # inherited layers, those tests might need to be re-verified or adapted.
    # However, given UnetWrapper's simplicity, this is unlikely.


if __name__ == '__main__':
    unittest.main()
