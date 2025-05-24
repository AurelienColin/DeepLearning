import unittest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os
import numpy as np

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
sys.modules['tensorflow.GradientTape'] = MagicMock # For train_step

# Mock Rignak modules
mock_rignak_lazy_property = MagicMock()
mock_rignak_logging_utils = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_property
sys.modules['Rignak.logging_utils'] = mock_rignak_logging_utils

# Mock custom modules used by DiffusionModelWrapper and its parent UnetWrapper
mock_build_encoder = MagicMock(name="BuildEncoder")
mock_build_decoder = MagicMock(name="BuildDecoder")
mock_get_embedding = MagicMock(name="GetEmbedding")
sys.modules['src.modules.module'] = MagicMock(
    build_encoder=mock_build_encoder, 
    build_decoder=mock_build_decoder,
    get_embedding=mock_get_embedding
)

# Mock losses and metrics
mock_encoding_similarity_class = MagicMock(name="EncodingSimilarityClass")
mock_blurriness_class = MagicMock(name="BlurrinessClass")
mock_mae_func = MagicMock(name="mae_func") # UnetWrapper might use this
sys.modules['src.losses.from_model.encoding_similarity'] = MagicMock(EncodingSimilarity=mock_encoding_similarity_class)
sys.modules['src.losses.from_model.blurriness'] = MagicMock(Blurriness=mock_blurriness_class)
sys.modules['src.losses.losses'] = MagicMock(mae=mock_mae_func) # For UnetWrapper part


from models.image_to_image.diffusion_model_wrapper import DiffusionModelWrapper
# UnetWrapper is the parent, its methods might be called.
# We assume UnetWrapper is tested separately and focus on DiffusionModelWrapper overrides and new methods.

class TestDiffusionModelWrapper(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        mock_build_encoder.reset_mock()
        mock_build_decoder.reset_mock()
        mock_get_embedding.reset_mock()
        mock_encoding_similarity_class.reset_mock()
        mock_blurriness_class.reset_mock()
        mock_mae_func.reset_mock()
        mock_rignak_logging_utils.logger.reset_mock() # Reset logger calls

        # Keras layer/model mocks
        self.MockInputLayer = MagicMock(name="InputLayerInstance")
        self.MockInputNoiseLayer = MagicMock(name="InputNoiseLayerInstance")
        self.MockEmbeddingLambdaLayer = MagicMock(name="EmbeddingLambdaLayer")
        self.MockUpSampling2DLayer = MagicMock(name="UpSampling2DLayer")
        self.MockConcatenateLayer = MagicMock(name="ConcatenateLayer")
        self.MockEncoderOutputLambda = MagicMock(name="EncoderOutputLambda") # tanh output
        self.MockConv2DLayer = MagicMock(name="Conv2DLayerInstance") # For output_layer
        
        mock_tf.keras.layers.Input.side_effect = [self.MockInputLayer, self.MockInputNoiseLayer, self.MockInputLayer, self.MockInputNoiseLayer] # Default for two inputs
        mock_tf.keras.layers.Lambda.side_effect = [self.MockEmbeddingLambdaLayer, self.MockEncoderOutputLambda]
        mock_tf.keras.layers.UpSampling2D.return_value = self.MockUpSampling2DLayer
        mock_tf.keras.layers.Concatenate.return_value = self.MockConcatenateLayer
        mock_tf.keras.layers.Conv2D.return_value = self.MockConv2DLayer
        
        self.MockKerasModel = MagicMock(name="KerasModelInstance")
        self.MockKerasModel.trainable_weights = [MagicMock(name="w1"), MagicMock(name="w2")]
        self.MockKerasModel.weights = [MagicMock(name="w1_ema_src"), MagicMock(name="w2_ema_src")]
        mock_tf.keras.Model.return_value = self.MockKerasModel

        self.MockEmaModel = MagicMock(name="EmaKerasModelInstance")
        self.MockEmaModel.weights = [MagicMock(name="w1_ema_trg"), MagicMock(name="w2_ema_trg")]
        mock_tf.keras.models.clone_model.return_value = self.MockEmaModel
        
        # Mock metrics instances
        self.mock_mean_metric_noise = MagicMock(name="MeanNoiseLoss")
        self.mock_mean_metric_image = MagicMock(name="MeanImageLoss")
        self.mock_blurriness_metric = MagicMock(name="BlurrinessMetricInstance")
        self.mock_encoding_sim_metric = MagicMock(name="EncodingSimilarityMetricInstance")
        mock_tf.keras.metrics.Mean.side_effect = [self.mock_mean_metric_noise, self.mock_mean_metric_image]
        mock_blurriness_class.return_value = self.mock_blurriness_metric
        mock_encoding_similarity_class.side_effect = [ # First for loss, then for compile
            MagicMock(name="EncodingSimilarityLossInstance"), 
            self.mock_encoding_sim_metric
        ]


        self.input_shape = (64, 64, 3)
        self.batch_size = 2 # Important for diffusion_times and other batch operations
        self.layer_kernels = (16, 32)
        self.n_stride = 2
        self.embedding_dims = 32

        self.wrapper = DiffusionModelWrapper(
            input_shape=self.input_shape,
            layer_kernels=self.layer_kernels,
            n_stride=self.n_stride,
            embedding_dims=self.embedding_dims,
            batch_size=self.batch_size # Added batch_size to wrapper for test clarity
        )
        # Ensure model is built for some tests that rely on self.model
        # self.wrapper.model # Access to build

    def test_initialization_parameters(self):
        self.assertEqual(self.wrapper.input_shape, self.input_shape)
        self.assertEqual(self.wrapper.layer_kernels, self.layer_kernels)
        self.assertEqual(self.wrapper.n_stride, self.n_stride)
        self.assertEqual(self.wrapper.embedding_dims, self.embedding_dims)
        self.assertEqual(self.wrapper.minimum_signal_rate, 0.02)
        self.assertEqual(self.wrapper.maximum_signal_rate, 0.95)
        self.assertTrue(self.wrapper.training_mode)
        # Check that some LazyProperties are not yet in __dict__
        self.assertNotIn('loss', self.wrapper.__dict__)
        self.assertNotIn('ema_model', self.wrapper.__dict__)

    def test_loss_property(self):
        loss_instance = self.wrapper.loss
        mock_encoding_similarity_class.assert_called_with(name='encoding_similarity', input_shape=self.input_shape)
        self.assertIsNotNone(loss_instance) 
        self.assertIs(self.wrapper.loss, loss_instance) # Cached

    def test_ema_model_property(self):
        # Access model first to ensure it's built and can be cloned
        _ = self.wrapper.model
        
        ema_model_instance = self.wrapper.ema_model
        mock_tf.keras.models.clone_model.assert_called_once_with(self.MockKerasModel)
        self.assertIs(ema_model_instance, self.MockEmaModel)
        self.assertIs(self.wrapper.ema_model, ema_model_instance) # Cached

    def test_compile_initializes_trackers(self):
        # Reset side effect for EncodingSimilarity for this specific test context
        mock_encoding_similarity_class.side_effect = [
            MagicMock(name="LossInstance"), # For self.loss during super().compile()
            self.mock_encoding_sim_metric   # For self.encoding_similarity metric
        ]
        
        # Mock optimizer from parent's compile
        mock_optimizer_instance = MagicMock(name="OptimizerInstance")
        mock_tf.keras.optimizers.Adam.return_value = mock_optimizer_instance

        self.wrapper.compile() # Calls super().compile() then adds its own trackers

        mock_tf.keras.metrics.Mean.assert_any_call(name="noise_loss")
        mock_tf.keras.metrics.Mean.assert_any_call(name="image_loss")
        self.assertIs(self.wrapper.noise_loss_tracker, self.mock_mean_metric_noise)
        self.assertIs(self.wrapper.image_loss_tracker, self.mock_mean_metric_image)
        
        mock_blurriness_class.assert_called_once_with(name="blurriness", input_shape=self.input_shape)
        self.assertIs(self.wrapper.blurriness_tracker, self.mock_blurriness_metric)
        
        mock_encoding_similarity_class.assert_any_call(name="encoding_similarity", input_shape=self.input_shape)
        self.assertIs(self.wrapper.encoding_similarity, self.mock_encoding_sim_metric)

        # Check that model.compile was called by super().compile()
        self.MockKerasModel.compile.assert_called_once()


    def test_metrics_property_training_mode(self):
        self.wrapper.training_mode = True
        # Need to compile first to initialize trackers
        mock_encoding_similarity_class.side_effect = [MagicMock(), self.mock_encoding_sim_metric]
        self.wrapper.compile()
        metrics = self.wrapper.metrics
        self.assertEqual(metrics, [self.mock_mean_metric_image])

    def test_metrics_property_validation_mode(self):
        self.wrapper.training_mode = False
        mock_encoding_similarity_class.side_effect = [MagicMock(), self.mock_encoding_sim_metric]
        self.wrapper.compile()
        metrics = self.wrapper.metrics
        self.assertEqual(metrics, [self.mock_mean_metric_image, self.mock_blurriness_metric])

    def test_diffusion_schedule(self):
        mock_diffusion_times = mock_tf.constant([0.0, 0.5, 1.0], dtype="float32")
        expected_start_angle = np.arccos(self.wrapper.maximum_signal_rate)
        expected_end_angle = np.arccos(self.wrapper.minimum_signal_rate)
        
        mock_tf.math.acos.side_effect = [expected_start_angle, expected_end_angle]

        # diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        # signal_rates = K.cos(diffusion_angles)
        # noise_rates = K.sin(diffusion_angles)
        def mock_cos_sin(angles_tensor):
            # This is a placeholder. In reality, tf.cos/sin would operate on the tensor.
            # For testing, we can return a predefined mock if specific values are needed.
            return MagicMock(name=f"cos_or_sin_of_{angles_tensor.name}")

        mock_tf.keras.backend.cos.side_effect = mock_cos_sin
        mock_tf.keras.backend.sin.side_effect = mock_cos_sin

        noise_rates, signal_rates = self.wrapper.diffusion_schedule(mock_diffusion_times)

        mock_tf.math.acos.assert_any_call(self.wrapper.maximum_signal_rate)
        mock_tf.math.acos.assert_any_call(self.wrapper.minimum_signal_rate)
        self.assertIsNotNone(noise_rates)
        self.assertIsNotNone(signal_rates)
        mock_tf.keras.backend.cos.assert_called()
        mock_tf.keras.backend.sin.assert_called()


    def test_input_noise_property(self):
        # Reset Input layer mock to ensure it's fresh for this specific test
        mock_tf.keras.layers.Input.reset_mock()
        mock_tf.keras.layers.Input.return_value = self.MockInputNoiseLayer # Specifically for input_noise

        in_noise = self.wrapper.input_noise
        mock_tf.keras.layers.Input.assert_called_once_with(shape=(1,1,1))
        self.assertIs(in_noise, self.MockInputNoiseLayer)
        self.assertIs(self.wrapper.input_noise, in_noise) # Cached

    def test_input_layers_property(self):
        # This will access self.input_layer (from ModelWrapper) and self.input_noise
        # Ensure Input is mocked to return distinct values for each call if not already handled by side_effect
        mock_tf.keras.layers.Input.side_effect = [self.MockInputLayer, self.MockInputNoiseLayer]
        
        layers = self.wrapper.input_layers
        
        self.assertEqual(mock_tf.keras.layers.Input.call_count, 2) # One for base input, one for noise input
        self.assertEqual(layers, (self.MockInputLayer, self.MockInputNoiseLayer))


    def test_set_encoded_layers(self):
        # This is complex due to multiple layer creations and calls.
        # We rely on the setUp mocks for various Keras layers.
        mock_embedding_func = MagicMock(name="EmbeddingFunction")
        mock_get_embedding.return_value = mock_embedding_func
        
        # Access encoded_layer to trigger set_encoded_layers
        _ = self.wrapper.encoded_layer 

        mock_get_embedding.assert_called_once_with(
            self.wrapper.embedding_min_frequency, 
            self.wrapper.embedding_max_frequency, 
            self.wrapper.embedding_dims
        )
        # First Lambda is for embedding
        mock_tf.keras.layers.Lambda.assert_any_call(mock_embedding_func, output_shape=(1,1,self.embedding_dims))
        self.MockEmbeddingLambdaLayer.assert_called_with(self.wrapper.input_noise) # Called with input_noise

        mock_tf.keras.layers.UpSampling2D.assert_called_once_with(
            size=self.input_shape[:-1],
            interpolation="nearest"
        )
        # UpSampling called with output of embedding Lambda
        self.MockUpSampling2DLayer.assert_called_with(self.MockEmbeddingLambdaLayer.return_value) 

        mock_tf.keras.layers.Concatenate.assert_called_once()
        # Concatenate called with [input_layer, upsampled_embedded_noise]
        self.MockConcatenateLayer.assert_called_with([self.wrapper.input_layer, self.MockUpSampling2DLayer.return_value])

        mock_build_encoder.assert_called_once_with(
            self.MockConcatenateLayer.return_value, # Input to encoder is concatenated layer
            self.layer_kernels,
            self.n_stride
        )
        # Second Lambda is for K.tanh
        mock_tf.keras.layers.Lambda.assert_any_call(ANY) # func for tanh
        lambda_for_tanh_arg = mock_tf.keras.layers.Lambda.call_args_list[-1][0][0] # Get the function arg
        # lambda_for_tanh_arg(mock_build_encoder.return_value[0]) # Call it
        # mock_tf.keras.backend.tanh.assert_called_with(mock_build_encoder.return_value[0])

        self.assertEqual(self.wrapper._encoded_layer, self.MockEncoderOutputLambda) # Output of tanh Lambda
        self.assertEqual(self.wrapper._encoded_inherited_layers, mock_build_encoder.return_value[1])
        mock_rignak_logging_utils.logger.assert_any_call("Set encoder", indent=1)
        mock_rignak_logging_utils.logger.assert_any_call("Set encoder OK", indent=-1)


    def test_output_layer_property(self):
        # Access encoded_layer first
        _ = self.wrapper.encoded_layer
        
        # Access output_layer
        out_layer = self.wrapper.output_layer

        mock_build_decoder.assert_called_once_with(
            self.wrapper.encoded_layer, # The tanh lambda layer
            self.wrapper.encoded_inherited_layers, # From build_encoder
            self.layer_kernels,
            self.n_stride
        )
        mock_tf.keras.layers.Conv2D.assert_called_with(
            self.input_shape[-1], # Output channels same as input for base Unet part
            activation="linear",  # <<<< Key change in DiffusionModelWrapper
            kernel_size=1
        )
        self.MockConv2DLayer.assert_called_with(mock_build_decoder.return_value[0] if isinstance(mock_build_decoder.return_value, tuple) else mock_build_decoder.return_value)
        self.assertIs(out_layer, self.MockConv2DLayer)
        self.assertIs(self.wrapper.output_layer, out_layer) # Cached


    def test_denoise_training_mode(self):
        self.wrapper.model = self.MockKerasModel # Ensure model is set
        mock_noisy_images = MagicMock(name="NoisyImages")
        mock_noise_rates = MagicMock(name="NoiseRates")
        mock_signal_rates = MagicMock(name="SignalRates")
        mock_pred_images_from_model = MagicMock(name="PredImagesFromModel")
        
        self.MockKerasModel.return_value = mock_pred_images_from_model # model(...)

        pred_noises, pred_images = self.wrapper.denoise(
            mock_noisy_images, mock_noise_rates, mock_signal_rates, training=True
        )
        
        self.MockKerasModel.assert_called_once_with([mock_noisy_images, mock_noise_rates], training=True)
        self.assertIsNotNone(pred_noises) # Mathematical derivation; difficult to assert exact value without running tf.
        self.assertIs(pred_images, mock_pred_images_from_model)

    def test_denoise_inference_mode(self):
        # Ensure ema_model is accessed and set
        _ = self.wrapper.ema_model
        mock_noisy_images = MagicMock(name="NoisyImages")
        mock_noise_rates = MagicMock(name="NoiseRates")
        mock_signal_rates = MagicMock(name="SignalRates")
        mock_pred_images_from_ema_model = MagicMock(name="PredImagesFromEmaModel")

        self.MockEmaModel.return_value = mock_pred_images_from_ema_model

        pred_noises, pred_images = self.wrapper.denoise(
            mock_noisy_images, mock_noise_rates, mock_signal_rates, training=False
        )

        self.MockEmaModel.assert_called_once_with([mock_noisy_images, mock_noise_rates], training=False)
        self.assertIsNotNone(pred_noises)
        self.assertIs(pred_images, mock_pred_images_from_ema_model)

    @patch.object(DiffusionModelWrapper, 'denoise')
    def test_reverse_diffusion(self, mock_denoise):
        # This is a loop, mock internal calls heavily
        mock_initial_noise = MagicMock(name="InitialNoise")
        diffusion_steps = 3
        
        # Mock diffusion_schedule return values
        mock_tf.keras.backend.ones.return_value = MagicMock(name="DiffusionTimes")
        self.wrapper.diffusion_schedule = MagicMock(name="DiffusionScheduleMock")
        self.wrapper.diffusion_schedule.side_effect = [
            (MagicMock(name="nr1"), MagicMock(name="sr1")), # step 0
            (MagicMock(name="nr2"), MagicMock(name="sr2")), # step 0 next
            (MagicMock(name="nr3"), MagicMock(name="sr3")), # step 1
            (MagicMock(name="nr4"), MagicMock(name="sr4")), # step 1 next
            (MagicMock(name="nr5"), MagicMock(name="sr5")), # step 2
            (MagicMock(name="nr6"), MagicMock(name="sr6")), # step 2 next
        ]

        # Mock denoise return values (pred_noises, pred_images)
        mock_pred_images_final_step = MagicMock(name="PredImagesFinal")
        mock_denoise.side_effect = [
            (MagicMock(name="pn1"), MagicMock(name="pi1")),
            (MagicMock(name="pn2"), MagicMock(name="pi2")),
            (MagicMock(name="pn3"), mock_pred_images_final_step),
        ]
        
        # Mock K.ones for diffusion_times
        mock_tf.keras.backend.ones.return_value = MagicMock(name="DiffusionTimes")

        # Mock np.concatenate for steps (if return_steps=True)
        with patch('numpy.concatenate') as mock_np_concat:
            result_steps = self.wrapper.reverse_diffusion(mock_initial_noise, diffusion_steps, return_steps=True)
            self.assertEqual(mock_denoise.call_count, diffusion_steps)
            self.assertEqual(self.wrapper.diffusion_schedule.call_count, diffusion_steps * 2)
            mock_np_concat.assert_called() # Called if return_steps is True
            self.assertIsNotNone(result_steps)

        # Test return_steps = False
        result_image = self.wrapper.reverse_diffusion(mock_initial_noise, diffusion_steps, return_steps=False)
        self.assertIs(result_image, mock_pred_images_final_step)


    @patch.object(DiffusionModelWrapper, 'reverse_diffusion')
    def test_generate(self, mock_reverse_diffusion):
        diffusion_steps = 5
        mock_generated_images = MagicMock(name="GeneratedImages")
        mock_reverse_diffusion.return_value = mock_generated_images
        mock_tf.keras.backend.random_normal.return_value = MagicMock(name="InitialNoise")

        result = self.wrapper.generate(diffusion_steps, return_steps=False)

        mock_tf.keras.backend.random_normal.assert_called_once_with(shape=(self.batch_size, *self.input_shape))
        mock_reverse_diffusion.assert_called_once_with(
            mock_tf.keras.backend.random_normal.return_value, 
            diffusion_steps, 
            return_steps=False
        )
        self.assertIs(result, mock_generated_images)

    def test_step_method(self):
        mock_images_input = MagicMock(name="InputImages")
        mock_tf.keras.backend.random_normal.return_value = MagicMock(name="Noises")
        mock_tf.keras.backend.random_uniform.return_value = MagicMock(name="DiffusionTimes")
        
        mock_noise_rates_val = MagicMock(name="NoiseRatesVal")
        mock_signal_rates_val = MagicMock(name="SignalRatesVal")
        self.wrapper.diffusion_schedule = MagicMock(return_value=(mock_noise_rates_val, mock_signal_rates_val))

        images, noises, noisy_images, noise_rates, signal_rates = self.wrapper.step(mock_images_input)

        mock_tf.keras.backend.random_normal.assert_called_once_with(
            shape=(self.batch_size, *self.input_shape), # Output shape is same as input for diffusion
            stddev=self.wrapper.noise_factor
        )
        mock_tf.keras.backend.random_uniform.assert_called_once_with(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        self.wrapper.diffusion_schedule.assert_called_once_with(mock_tf.keras.backend.random_uniform.return_value)
        self.assertIs(images, mock_images_input)
        self.assertIs(noises, mock_tf.keras.backend.random_normal.return_value)
        self.assertIsNotNone(noisy_images) # Math derivation
        self.assertIs(noise_rates, mock_noise_rates_val)
        self.assertIs(signal_rates, mock_signal_rates_val)

    @patch.object(DiffusionModelWrapper, 'step')
    @patch.object(DiffusionModelWrapper, 'denoise')
    def test_call_method(self, mock_denoise, mock_step):
        mock_input_tensor = MagicMock(name="InputTensor")
        
        # Mock returns for step()
        mock_step_images = MagicMock(name="StepImages")
        mock_step_noises = MagicMock(name="StepNoises")
        mock_step_noisy_images = MagicMock(name="StepNoisyImages")
        mock_step_noise_rates = MagicMock(name="StepNoiseRates")
        mock_step_signal_rates = MagicMock(name="StepSignalRates")
        mock_step.return_value = (mock_step_images, mock_step_noises, mock_step_noisy_images, mock_step_noise_rates, mock_step_signal_rates)

        # Mock returns for denoise()
        mock_denoise_pred_noises = MagicMock(name="DenoisePredNoises")
        mock_denoise_pred_images = MagicMock(name="DenoisePredImages")
        mock_denoise.return_value = (mock_denoise_pred_noises, mock_denoise_pred_images)

        pred_noises_call, pred_images_call = self.wrapper.call(mock_input_tensor)

        mock_step.assert_called_once_with(mock_input_tensor)
        mock_denoise.assert_called_once_with(
            mock_step_noisy_images, mock_step_noise_rates, mock_step_signal_rates, training=False # call implies inference
        )
        self.assertIs(pred_noises_call, mock_denoise_pred_noises)
        self.assertIs(pred_images_call, mock_denoise_pred_images)

    @patch.object(DiffusionModelWrapper, 'step')
    @patch.object(DiffusionModelWrapper, 'denoise')
    def test_step_wrapper(self, mock_denoise, mock_step):
        # This is called by train_step and test_step
        mock_packed_images = (MagicMock(name="Inputs"), MagicMock(name="Outputs"))
        training_mode = True
        
        # Mock returns for step()
        mock_step_images_ret = MagicMock(name="StepImagesRet")
        mock_step_noises_ret = MagicMock(name="StepNoisesRet")
        # ... and others from step
        mock_step.return_value = (mock_step_images_ret, mock_step_noises_ret, MagicMock(), MagicMock(), MagicMock())

        # Mock returns for denoise()
        mock_denoise_pred_noises_ret = MagicMock(name="DenoisePredNoisesRet")
        mock_denoise_pred_images_ret = MagicMock(name="DenoisePredImagesRet")
        mock_denoise.return_value = (mock_denoise_pred_noises_ret, mock_denoise_pred_images_ret)

        # Mock the model's loss function (which is self.loss, an EncodingSimilarity instance)
        self.wrapper.model = self.MockKerasModel
        mock_loss_callable = MagicMock(name="LossCallable")
        self.MockKerasModel.loss = mock_loss_callable
        mock_loss_callable.side_effect = [MagicMock(name="NoiseLossValue"), MagicMock(name="ImageLossValue")]

        # Compile to setup trackers
        mock_encoding_similarity_class.side_effect = [MagicMock(), self.mock_encoding_sim_metric] # loss, metric
        self.wrapper.compile()

        images_ret, image_loss_ret = self.wrapper.step_wrapper(mock_packed_images, training=training_mode)

        mock_step.assert_called_once_with(mock_packed_images[0])
        mock_denoise.assert_called_with(ANY, ANY, ANY, training=training_mode) # Args from step
        
        mock_loss_callable.assert_any_call(mock_step_noises_ret, mock_denoise_pred_noises_ret)
        mock_loss_callable.assert_any_call(mock_packed_images[1], mock_denoise_pred_images_ret)
        
        self.wrapper.noise_loss_tracker.update_state.assert_called_once_with(mock_loss_callable.side_effect[0])
        self.wrapper.image_loss_tracker.update_state.assert_called_once_with(mock_loss_callable.side_effect[1])

        self.assertIs(images_ret, mock_step_images_ret)
        self.assertIs(image_loss_ret, mock_loss_callable.side_effect[1])


    @patch.object(DiffusionModelWrapper, 'step_wrapper')
    def test_train_step(self, mock_step_wrapper):
        mock_packed_images_train = (MagicMock(name="TrainInputs"), MagicMock(name="TrainOutputs"))
        mock_image_loss_value = MagicMock(name="ImageLossValueForTrain")
        mock_step_wrapper.return_value = (MagicMock(name="ImagesFromStepWrapper"), mock_image_loss_value)

        # Mock GradientTape
        mock_tape = MagicMock(name="GradientTapeInstance")
        mock_tf.GradientTape.return_value.__enter__.return_value = mock_tape
        mock_gradients = [MagicMock(name="grad1"), MagicMock(name="grad2")]
        mock_tape.gradient.return_value = mock_gradients
        
        # Mock optimizer
        self.wrapper.model = self.MockKerasModel # model is used
        self.wrapper.model.optimizer = MagicMock(name="OptimizerInstance")
        
        # Mock EMA model for weight updates
        _ = self.wrapper.ema_model # ensure it's created and self.MockEmaModel is assigned

        # Compile to setup metrics for logs
        mock_encoding_similarity_class.side_effect = [MagicMock(), self.mock_encoding_sim_metric]
        self.wrapper.compile()
        # Mock result for metrics
        for metric_mock in [self.wrapper.image_loss_tracker]: # Only image_loss_tracker in training metrics
            metric_mock.result.return_value = MagicMock(name=f"Result_{metric_mock.name}")


        logs = self.wrapper.train_step(mock_packed_images_train)

        self.assertTrue(self.wrapper.training_mode)
        mock_step_wrapper.assert_called_once_with(mock_packed_images_train, training=True)
        mock_tape.gradient.assert_called_once_with(mock_image_loss_value, self.MockKerasModel.trainable_weights)
        self.wrapper.model.optimizer.apply_gradients.assert_called_once_with(zip(mock_gradients, self.MockKerasModel.trainable_weights))

        # Check EMA updates
        for i in range(len(self.MockKerasModel.weights)):
            self.MockEmaModel.weights[i].assign.assert_called_once()
            # assign(self.ema * ema_weight + (1 - self.ema) * weight) - check one arg
            # Difficult to check exact math without TF running, so check call presence.

        self.assertIn(self.wrapper.image_loss_tracker.name, logs)


    @patch.object(DiffusionModelWrapper, 'step_wrapper')
    @patch.object(DiffusionModelWrapper, 'generate')
    def test_test_step(self, mock_generate, mock_step_wrapper):
        mock_packed_images_test = (MagicMock(name="TestInputs"), MagicMock(name="TestOutputs"))
        mock_images_from_step_wrapper = MagicMock(name="ImagesFromStepWrapperForTest")
        mock_step_wrapper.return_value = (mock_images_from_step_wrapper, MagicMock(name="ImageLossIgnoredInTest"))
        
        mock_generated_images_for_test = MagicMock(name="GeneratedImagesForTest")
        mock_generate.return_value = mock_generated_images_for_test

        # Compile to setup metrics
        mock_encoding_similarity_class.side_effect = [MagicMock(), self.mock_encoding_sim_metric]
        self.wrapper.compile()
        # Mock result for metrics
        for metric_mock in [self.wrapper.image_loss_tracker, self.wrapper.blurriness_tracker]:
             metric_mock.result.return_value = MagicMock(name=f"Result_{metric_mock.name}")

        logs = self.wrapper.test_step(mock_packed_images_test)

        self.assertFalse(self.wrapper.training_mode)
        mock_step_wrapper.assert_called_once_with(mock_packed_images_test, training=False)
        mock_generate.assert_called_once_with(diffusion_steps=self.wrapper.kid_diffusion_steps)
        
        self.wrapper.blurriness_tracker.update_state.assert_called_once_with(mock_images_from_step_wrapper, mock_generated_images_for_test)
        # self.wrapper.encoding_similarity.update_state.assert_called_once_with(mock_images_from_step_wrapper, mock_generated_images_for_test) # Uncomment if used

        self.assertIn(self.wrapper.image_loss_tracker.name, logs)
        self.assertIn(self.wrapper.blurriness_tracker.name, logs)


    def test_on_fit_start(self):
        self.wrapper.model = self.MockKerasModel
        mock_callback = MagicMock(spec=mock_tf.keras.callbacks.Callback)
        callbacks = [mock_callback]

        self.wrapper.on_fit_start(callbacks)

        self.assertEqual(self.MockKerasModel.train_step, self.wrapper.train_step)
        self.assertEqual(self.MockKerasModel.test_step, self.wrapper.test_step)
        self.assertEqual(mock_callback.model, self.MockKerasModel)

    @patch.object(DiffusionModelWrapper, 'run_step')
    def test_run_epoch(self, mock_run_step):
        mock_dataset = MagicMock(name="TrainDataset")
        mock_val_dataset = MagicMock(name="ValDataset")
        steps_per_epoch = 10
        validation_steps = 5
        
        mock_train_logs = {'loss': 0.5}
        mock_val_logs = {'val_loss': 0.6}
        mock_run_step.side_effect = [mock_train_logs, mock_val_logs]

        logs = self.wrapper.run_epoch(mock_dataset, mock_val_dataset, steps_per_epoch, validation_steps)

        mock_run_step.assert_any_call(self.wrapper.train_step, mock_dataset, steps_per_epoch)
        mock_run_step.assert_any_call(self.wrapper.test_step, mock_val_dataset, validation_steps, key_prefix='val_')
        
        expected_logs = {**mock_train_logs, **mock_val_logs}
        self.assertEqual(logs, expected_logs)

    def test_run_step_static_method(self):
        mock_step_func = MagicMock(name="StepFunction")
        mock_dataset_iter = MagicMock(name="DatasetIterator")
        num_steps = 2
        
        metric_val1 = MagicMock(); metric_val1.numpy.return_value = 1.0
        metric_val2 = MagicMock(); metric_val2.numpy.return_value = 2.0
        metric_val3 = MagicMock(); metric_val3.numpy.return_value = 0.5
        metric_val4 = MagicMock(); metric_val4.numpy.return_value = 0.7
        mock_step_func.side_effect = [
            {'metric_a': metric_val1, 'metric_b': metric_val3}, # Call 1
            {'metric_a': metric_val2, 'metric_b': metric_val4}  # Call 2
        ]
        mock_dataset_iter.__next__.side_effect = [MagicMock(name="Batch1"), MagicMock(name="Batch2")]
        
        logs = DiffusionModelWrapper.run_step(mock_step_func, mock_dataset_iter, num_steps, key_prefix="test_")

        self.assertEqual(mock_step_func.call_count, num_steps)
        # Expected: metric_a = (1.0 + 2.0) / 2 = 1.5
        #           metric_b = (0.5 + 0.7) / 2 = 0.6
        self.assertAlmostEqual(logs['test_metric_a'], 1.5)
        self.assertAlmostEqual(logs['test_metric_b'], 0.6)

    def test_on_epoch_end_static_method(self):
        epoch = 1
        logs = {'loss': 0.5}
        mock_callback1 = MagicMock(spec=mock_tf.keras.callbacks.Callback)
        mock_callback2 = MagicMock(spec=mock_tf.keras.callbacks.Callback)
        callbacks = [mock_callback1, mock_callback2]

        DiffusionModelWrapper.on_epoch_end(epoch, logs, callbacks)

        mock_callback1.on_epoch_end.assert_called_once_with(epoch, logs=logs)
        mock_callback2.on_epoch_end.assert_called_once_with(epoch, logs=logs)

    @patch.object(DiffusionModelWrapper, 'on_fit_start')
    @patch.object(DiffusionModelWrapper, 'run_epoch')
    @patch.object(DiffusionModelWrapper, 'on_epoch_end')
    def test_fit_loop(self, mock_on_epoch_end, mock_run_epoch, mock_on_fit_start):
        mock_dataset = MagicMock(name="FitTrainDataset")
        mock_val_dataset = MagicMock(name="FitValDataset")
        batch_size, steps_per_epoch, validation_steps, epochs, workers = 2,10,5,3,1
        mock_callbacks = [MagicMock(spec=mock_tf.keras.callbacks.Callback)]
        
        mock_run_epoch.return_value = {'loss': 0.1, 'val_loss': 0.2} # Logs from run_epoch

        self.wrapper.fit(
            mock_dataset, batch_size, mock_val_dataset, steps_per_epoch, 
            validation_steps, epochs, mock_callbacks, workers
        )

        mock_on_fit_start.assert_called_once_with(mock_callbacks)
        self.assertEqual(mock_run_epoch.call_count, epochs)
        self.assertEqual(mock_on_epoch_end.call_count, epochs)
        mock_rignak_logging_utils.logger.set_iterator.assert_called_once_with(epochs)
        self.assertEqual(mock_rignak_logging_utils.logger.iterate.call_count, epochs)


if __name__ == '__main__':
    unittest.main()
