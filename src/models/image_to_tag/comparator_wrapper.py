import typing
from dataclasses import dataclass

import tensorflow as tf
from rignak.src.lazy_property import LazyProperty
import tensorflow.keras.backend as K

from src.losses import losses
from src.models.model_wrapper import ModelWrapper
from src.modules.module import build_decoder

from src.config import DEFAULT_ACTIVATION


@dataclass
class Comparator(ModelWrapper):
    layer_kernels: typing.Sequence[int] = (32, 48, 64, 96, 128)
    levels: int = 2

    _encoder: typing.Optional[tf.keras.models.Model] = None
    _decoder: typing.Optional[tf.keras.models.Model] = None
    _post_encoder: typing.Optional[tf.keras.models.Model] = None

    _input_layers: typing.Optional[typing.Sequence[tf.keras.layers.Layer]] = None
    _post_encoded_layer: typing.Optional[tf.keras.layers.Layer] = None
    _decoded_layer: typing.Optional[tf.keras.layers.Layer] = None
    
    _encoded_input_layer: typing.Optional[tf.keras.layers.Layer] = None

    @LazyProperty
    def input_layers(self) -> typing.Sequence[tf.keras.layers.Layer]:
        return tf.keras.layers.Input(shape=(2, *self.input_shape)),

    @LazyProperty
    def loss(self) -> typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        comparative_loss = losses.Loss((losses.cross_entropy,))
        decoder_loss = losses.Loss((losses.mae,))
        return comparative_loss, decoder_loss, decoder_loss

    @LazyProperty
    def metrics(self) -> typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        return ()

    @LazyProperty
    def encoder(self) -> tf.keras.models.Model:
        return tf.keras.Model(inputs=self.input_layer, outputs=self.encoded_layer, name="shared_encoder")
        
    
    @LazyProperty
    def encoded_input_layer(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Input(shape=self.encoded_layer.shape[1:])

    @LazyProperty
    def post_encoded_layer(self) -> tf.keras.layers.Layer:
        # current_layer = tf.keras.layers.GlobalAveragePooling2D()(self.encoded_layer)
        current_layer = tf.keras.layers.GlobalAveragePooling2D()(self.encoded_input_layer)
        post_encoded_layer = tf.keras.layers.Dense(self.layer_kernels[-1], activation=DEFAULT_ACTIVATION)(current_layer)
        return post_encoded_layer

    @LazyProperty
    def post_encoder(self) -> tf.keras.models.Model:
        return tf.keras.Model(
            # inputs=self.encoded_layer, 
            inputs=self.encoded_input_layer, 
            outputs=self.post_encoded_layer, 
            name="shared_post_encoder"
        )

    @LazyProperty
    def decoded_layer(self) -> tf.keras.layers.Layer:
        current_layer = build_decoder(
            self.encoded_input_layer,
            # self.encoded_layer,
            (),
            self.layer_kernels,
            superseeded_conv_layer=self.superseeded_conv_layer,
            superseeded_conv_kwargs=self.superseeded_conv_kwargs,
        )
        decoded_layer = tf.keras.layers.Conv2D(
            self.input_shape[-1],
            activation="sigmoid",
            kernel_size=1
        )(current_layer)
        return decoded_layer

    @LazyProperty
    def decoder(self) -> tf.keras.models.Model:
        return tf.keras.Model(
            # inputs=self.encoded_layer, 
            inputs=self.encoded_input_layer, 
            outputs=self.decoded_layer,
             name="shared_decoder"
             )

    @LazyProperty
    def output_layer(self) -> typing.Sequence[tf.keras.layers.Layer]:
        input_layer1 = tf.keras.layers.Lambda(lambda x: x[:, 0])(self.input_layers[0])
        input_layer2 = tf.keras.layers.Lambda(lambda x: x[:, 1])(self.input_layers[0])

        encoded_a = self.encoder(input_layer1)
        encoded_b = self.encoder(input_layer2)

        post_encoded_a = self.post_encoder(encoded_a)
        post_encoded_b = self.post_encoder(encoded_b)

        decoded_a = self.decoder(encoded_a)
        decoded_b = self.decoder(encoded_b)

        distance = tf.keras.layers.Lambda(
            lambda tensors: tf.math.abs(tensors[0] - tensors[1])
        )([post_encoded_a, post_encoded_b])

        prediction = tf.keras.layers.Dense(self.levels, activation='sigmoid')(distance)
        return prediction, decoded_a, decoded_b
