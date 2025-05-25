import unittest
import tensorflow as tf
import typing

# Import modules to be tested
from src.modules.module import get_embedding, build_encoder, build_decoder
# The following are used by build_encoder/decoder indirectly, but not directly tested here
# from src.modules.blocks.convolution_block import ConvolutionBlock
# from src.modules.blocks.deconvolution_block import DeconvolutionBlock

class TestModuleFunctions(unittest.TestCase):
    def test_get_embedding(self) -> None:
        embedding_min_frequency: float = 1.0
        embedding_max_frequency: float = 1000.0
        embedding_dims: int = 64
        
        embedding_fn = get_embedding(embedding_min_frequency, embedding_max_frequency, embedding_dims)
        self.assertTrue(callable(embedding_fn))
        
        batch_size: int = 2
        input_tensor: tf.Tensor = tf.random.normal([batch_size, 1, 1, 1])
        embedded_tensor = embedding_fn(input_tensor)
        
        self.assertIsInstance(embedded_tensor, tf.Tensor)
        self.assertEqual(embedded_tensor.shape, (batch_size, 1, 1, embedding_dims))

    def test_build_encoder(self) -> None:
        height: int = 64
        width: int = 64
        channels: int = 3
        input_layer = tf.keras.Input(shape=(height, width, channels))
        
        layer_kernels: typing.Sequence[int] = [16, 32, 64]
        n_stride: int = 1

        output_layer, inherited_layers = build_encoder(input_layer, layer_kernels, n_stride)
        
        self.assertIsInstance(output_layer, tf.keras.KerasTensor)
        self.assertIsInstance(inherited_layers, list)
        self.assertEqual(len(inherited_layers), len(layer_kernels))

        expected_height = height // (2**len(layer_kernels))
        expected_width = width // (2**len(layer_kernels))
        expected_channels = layer_kernels[-1]
        
        self.assertEqual(list(output_layer.shape), [None, expected_height, expected_width, expected_channels])
        for i, inherited_layer in enumerate(inherited_layers):
            self.assertIsInstance(inherited_layer, tf.keras.KerasTensor)
            self.assertEqual(list(inherited_layer.shape), [None, height // (2**i), width // (2**i), layer_kernels[i]])

    def test_build_decoder(self) -> None:
        start_height: int = 8 
        start_width: int = 8
        start_channels: int = 64
        current_layer_input = tf.keras.Input(shape=(start_height, start_width, start_channels), name="decoder_input")
        
        encoder_layer_kernels: typing.Sequence[int] = [16, 32, 64] 
        decoder_layer_kernels: typing.Sequence[int] = encoder_layer_kernels[::-1] # [64, 32, 16]
        n_stride: int = 1

        mock_inherited_layers: typing.List[tf.keras.KerasTensor] = []
        model_input_h = start_height * (2**len(encoder_layer_kernels))
        model_input_w = start_width * (2**len(encoder_layer_kernels))

        for i in range(len(encoder_layer_kernels)):
            h_inherited = model_input_h // (2**i)
            w_inherited = model_input_w // (2**i)
            c_inherited = encoder_layer_kernels[i]
            shape_inherited = (h_inherited, w_inherited, c_inherited)
            mock_inherited_layers.append(tf.keras.Input(shape=shape_inherited, name=f"inherited_{i}"))

        output_layer = build_decoder(current_layer_input, mock_inherited_layers, decoder_layer_kernels, n_stride)
        
        self.assertIsInstance(output_layer, tf.keras.KerasTensor)
        expected_height = start_height * (2**len(decoder_layer_kernels))
        expected_width = start_width * (2**len(decoder_layer_kernels))
        expected_channels = decoder_layer_kernels[0] 
        self.assertEqual(list(output_layer.shape), [None, expected_height, expected_width, expected_channels])

    def test_build_decoder_no_inherited_layers(self) -> None:
        start_height: int = 8 
        start_width: int = 8
        start_channels: int = 64
        decoder_layer_kernels: typing.Sequence[int] = [64, 32, 16]
        n_stride: int = 1

        current_layer_input_no_skip = tf.keras.Input(shape=(start_height, start_width, start_channels), name="decoder_input_no_skip")
        output_layer_no_skip = build_decoder(current_layer_input_no_skip, [], decoder_layer_kernels, n_stride)
        
        self.assertIsInstance(output_layer_no_skip, tf.keras.KerasTensor)
        expected_height = start_height * (2**len(decoder_layer_kernels))
        expected_width = start_width * (2**len(decoder_layer_kernels))
        expected_channels = decoder_layer_kernels[0]
        self.assertEqual(list(output_layer_no_skip.shape), [None, expected_height, expected_width, expected_channels])

if __name__ == '__main__':
    unittest.main()
