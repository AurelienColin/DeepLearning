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


from models.image_to_image.auto_encoder_wrapper import AutoEncoderWrapper
from losses.losses import Loss # This will use the mocked tf

# Mock custom modules used by AutoEncoderWrapper
mock_build_encoder = MagicMock()
mock_build_decoder = MagicMock()
sys.modules['src.modules.module'] = MagicMock(build_encoder=mock_build_encoder, build_decoder=mock_build_decoder)

mock_edge_loss_func = MagicMock(name='edge_loss_func')
mock_mae_func = MagicMock(name='mae_func')
sys.modules['src.losses.losses'] = MagicMock(edge_loss=mock_edge_loss_func, mae=mock_mae_func, Loss=Loss)


class TestAutoEncoderWrapper(unittest.TestCase):

    def setUp(self):
        # Reset mocks for each test to ensure independence
        mock_tf.reset_mock()
        mock_build_encoder.reset_mock()
        mock_build_decoder.reset_mock()
        mock_edge_loss_func.reset_mock()
        mock_mae_func.reset_mock()

        # Mock Keras layers and Model
        self.MockInputLayer = MagicMock(name="InputLayer")
        self.MockLambdaLayer = MagicMock(name="LambdaLayer")
        self.MockConv2DLayer = MagicMock(name="Conv2DLayer")
        
        mock_tf.keras.layers.Input.return_value = self.MockInputLayer
        mock_tf.keras.layers.Lambda.return_value = self.MockLambdaLayer
        mock_tf.keras.layers.Conv2D.return_value = self.MockConv2DLayer
        
        self.MockKerasModel = MagicMock(name="KerasModelInstance")
        mock_tf.keras.Model.return_value = self.MockKerasModel

        # Mock return values for encoder/decoder builders
        self.mock_encoder_output_layer = MagicMock(name="EncoderOutput")
        self.mock_encoder_inherited_layers = [MagicMock(name="EncInherited1"), MagicMock(name="EncInherited2")]
        mock_build_encoder.return_value = (self.mock_encoder_output_layer, self.mock_encoder_inherited_layers)

        self.mock_decoder_output_layer = MagicMock(name="DecoderOutput")
        mock_build_decoder.return_value = self.mock_decoder_output_layer
        
        # Mock K.tanh
        self.mock_ktanh_output = MagicMock(name="KTanhOutput")
        mock_tf.keras.backend.tanh.return_value = self.mock_ktanh_output

        self.input_shape = (128, 128, 3)
        self.output_shape_rgb = (128, 128, 3)
        self.output_shape_mono = (128, 128, 1)
        self.layer_kernels = (32, 64)
        self.n_stride = 2

    def create_wrapper(self, output_shape, **kwargs):
        params = {
            'input_shape': self.input_shape,
            'output_shape': output_shape,
            'layer_kernels': self.layer_kernels,
            'n_stride': self.n_stride,
            **kwargs
        }
        return AutoEncoderWrapper(**params)

    def test_initialization_parameters(self):
        wrapper = self.create_wrapper(self.output_shape_rgb)
        self.assertEqual(wrapper.input_shape, self.input_shape)
        self.assertEqual(wrapper.output_shape, self.output_shape_rgb)
        self.assertEqual(wrapper.layer_kernels, self.layer_kernels)
        self.assertEqual(wrapper.n_stride, self.n_stride)
        self.assertIsNone(wrapper._output_layer) # Lazy property not yet accessed
        self.assertIsNone(wrapper._encoded_layer)
        self.assertIsNone(wrapper._encoded_inherited_layers) # This is different from the property

    def test_set_encoded_layers(self):
        wrapper = self.create_wrapper(self.output_shape_rgb)
        
        # Call the method that uses build_encoder and K.tanh
        # This is usually called via the LazyProperty 'encoded_layer'
        _ = wrapper.encoded_layer # Access to trigger set_encoded_layers

        # Assertions
        mock_build_encoder.assert_called_once_with(
            wrapper.input_layer, # This is created in ModelWrapper's __post_init__
            self.layer_kernels,
            self.n_stride
        )
        mock_tf.keras.layers.Lambda.assert_called_once()
        # Check that K.tanh was called with the output of build_encoder
        lambda_func_arg = mock_tf.keras.layers.Lambda.call_args[0][0]
        self.assertEqual(lambda_func_arg(self.mock_encoder_output_layer), self.mock_ktanh_output)
        
        self.assertEqual(wrapper._encoded_layer, self.MockLambdaLayer) # Output of Lambda layer
        self.assertEqual(wrapper._encoded_inherited_layers, self.mock_encoder_inherited_layers)

    def test_encoded_layer_property(self):
        wrapper = self.create_wrapper(self.output_shape_rgb)
        # Access the property multiple times
        el1 = wrapper.encoded_layer
        el2 = wrapper.encoded_layer
        
        self.assertIs(el1, self.MockLambdaLayer)
        self.assertIs(el2, el1) # Should return the same cached instance
        # set_encoded_layers (and thus build_encoder) should only be called once
        mock_build_encoder.assert_called_once()

    def test_encoded_inherited_layers_property_is_empty_list(self):
        # This property in AutoEncoderWrapper specifically returns [],
        # different from _encoded_inherited_layers internal var.
        wrapper = self.create_wrapper(self.output_shape_rgb)
        # Populate the internal variable by accessing encoded_layer
        _ = wrapper.encoded_layer 
        self.assertIsNotNone(wrapper._encoded_inherited_layers) # Internal var is set
        
        # Access the public property
        public_inherited_layers = wrapper.encoded_inherited_layers
        self.assertEqual(public_inherited_layers, [])


    def test_output_layer_property(self):
        wrapper = self.create_wrapper(self.output_shape_rgb)
        
        # Access encoded_layer first to ensure those parts are built
        mock_encoded_layer_instance = wrapper.encoded_layer 
        # The public property encoded_inherited_layers returns []
        mock_encoded_inherited_layers_for_decoder = wrapper.encoded_inherited_layers

        # Access the output_layer property
        ol1 = wrapper.output_layer
        ol2 = wrapper.output_layer

        # Assertions
        mock_build_decoder.assert_called_once_with(
            mock_encoded_layer_instance,
            mock_encoded_inherited_layers_for_decoder, # Should be []
            self.layer_kernels,
            self.n_stride
        )
        mock_tf.keras.layers.Conv2D.assert_called_once_with(
            self.output_shape_rgb[-1], # Number of output channels
            activation="sigmoid",
            kernel_size=1
        )
        # Check that Conv2D was called with the output of build_decoder
        self.assertEqual(mock_tf.keras.layers.Conv2D.call_args[0][0], self.mock_decoder_output_layer)
        
        self.assertIs(ol1, self.MockConv2DLayer) # Output of Conv2D
        self.assertIs(ol2, ol1) # Cached
        mock_build_decoder.assert_called_once() # Called only once


    def test_build_model(self):
        wrapper = self.create_wrapper(self.output_shape_rgb)
        
        # Mock the input and output layers that build_model will retrieve
        wrapper._input_layer = self.MockInputLayer # Manually set for this test
        wrapper._output_layer = self.MockConv2DLayer # Manually set

        model = wrapper.build_model()

        mock_tf.keras.Model.assert_called_once_with(
            inputs=self.MockInputLayer,
            outputs=self.MockConv2DLayer,
            name="AutoEncoder" # Default name from ModelWrapper
        )
        self.assertIs(model, self.MockKerasModel)
        self.assertIs(wrapper.model, self.MockKerasModel) # Also check if assigned to property

    def test_loss_selection_rgb(self):
        wrapper = self.create_wrapper(self.output_shape_rgb) # 3 channels
        loss_func = wrapper.loss
        
        self.assertIsInstance(loss_func, Loss)
        self.assertEqual(loss_func.losses, (mock_edge_loss_func, mock_mae_func))
        self.assertEqual(loss_func.loss_weights, (0.1, 1))

    def test_loss_selection_rgba(self):
        wrapper = self.create_wrapper((128,128,4)) # 4 channels
        loss_func = wrapper.loss
        
        self.assertIsInstance(loss_func, Loss)
        self.assertEqual(loss_func.losses, (mock_edge_loss_func, mock_mae_func))
        self.assertEqual(loss_func.loss_weights, (0.1, 1))

    def test_loss_selection_mono(self):
        wrapper = self.create_wrapper(self.output_shape_mono) # 1 channel
        loss_func = wrapper.loss
        self.assertIs(loss_func, mock_mae_func)

    def test_metrics_selection_rgb(self):
        wrapper = self.create_wrapper(self.output_shape_rgb)
        metrics = wrapper.metrics
        self.assertEqual(metrics, (mock_edge_loss_func, mock_mae_func))

    def test_metrics_selection_mono(self):
        wrapper = self.create_wrapper(self.output_shape_mono)
        metrics = wrapper.metrics
        self.assertEqual(metrics, (mock_mae_func,))

    def test_desactivate_edge_loss(self):
        wrapper = self.create_wrapper(self.output_shape_rgb)
        # Access loss first to cache it (it would be the combined loss)
        _ = wrapper.loss 
        
        wrapper.desactivate_edge_loss()
        
        # Now, the loss property should return only mae
        self.assertIs(wrapper.loss, mock_mae_func)
        # Check if the internal _loss attribute was set
        self.assertIs(wrapper._loss, mock_mae_func)

    def test_compile_model(self):
        # Mostly testing the ModelWrapper's compile, but ensuring AE's loss/metrics are used
        wrapper = self.create_wrapper(self.output_shape_rgb)
        wrapper.model = self.MockKerasModel # Assign a mock model instance
        
        # Mock optimizer
        mock_optimizer_instance = MagicMock()
        mock_tf.keras.optimizers.Adam.return_value = mock_optimizer_instance

        wrapper.compile(learning_rate=0.001)

        mock_tf.keras.optimizers.Adam.assert_called_once_with(learning_rate=0.001)
        
        # Expected loss is the combined one for RGB
        expected_loss_obj = Loss((mock_edge_loss_func, mock_mae_func), loss_weights=(0.1,1))
        
        # The actual loss object passed to compile will be a new instance of Loss
        # We need to check the arguments passed to the model's compile method
        args, kwargs = self.MockKerasModel.compile.call_args
        
        compiled_loss = kwargs['loss']
        self.assertIsInstance(compiled_loss, Loss)
        self.assertEqual(compiled_loss.losses, expected_loss_obj.losses)
        self.assertEqual(compiled_loss.loss_weights, expected_loss_obj.loss_weights)
        
        self.assertEqual(kwargs['optimizer'], mock_optimizer_instance)
        self.assertEqual(kwargs['metrics'], wrapper.metrics)


if __name__ == '__main__':
    unittest.main()
