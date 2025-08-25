import typing
import warnings
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger

from src.generators.base_generators import BatchGenerator
from src.losses.from_model.blurriness import Blurriness
from src.losses.from_model.encoding_similarity import EncodingSimilarity
from src.models.image_to_image.unet_wrapper import UnetWrapper
from src.modules.module import get_embedding, build_encoder, build_decoder

Array_or_Tensor = typing.Union[np.ndarray, tf.Tensor]


@dataclass
class DiffusionModelWrapper(UnetWrapper):
    minimum_signal_rate: float = 0.02
    maximum_signal_rate: float = 0.95

    noise_factor: float = 2.
    embedding_min_frequency: float = 1.
    embedding_max_frequency: float = 1000.0
    embedding_dims: int = 32

    ema: float = 0.999  # Exponential Moving Average

    kid_diffusion_steps: int = 5
    noise_loss_tracker: typing.Optional[tf.keras.metrics.Metric] = None
    image_loss_tracker: typing.Optional[tf.keras.metrics.Metric] = None
    # kid: typing.Optional[tf.keras.metrics.Metric] = None
    encoding_similarity: typing.Optional[tf.keras.metrics.Metric] = None
    blurriness_tracker: typing.Optional[tf.keras.metrics.Metric] = None

    _input_noise: typing.Optional[tf.keras.layers.Layer] = None
    _ema_model: typing.Optional[tf.keras.models.Model] = None
    training_mode: bool = True

    @LazyProperty
    def loss(self) -> typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        return EncodingSimilarity(name='encoding_similarity', input_shape=self.input_shape)

    @LazyProperty
    def ema_model(self) -> tf.keras.models.Model:
        return tf.keras.models.clone_model(self.model)

    def compile(self) -> None:
        super().compile()
        self.noise_loss_tracker = tf.keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="image_loss")
        # self.kid = KID(name="kid", input_shape=self.input_shape)
        self.blurriness_tracker = Blurriness(name="blurriness", input_shape=self.input_shape)
        self.encoding_similarity = EncodingSimilarity(name="encoding_similarity", input_shape=self.input_shape)

    @property
    def metrics(self) -> typing.Sequence[tf.keras.metrics.Metric]:
        if self.training_mode:
            return [self.image_loss_tracker]
        else:
            return [self.image_loss_tracker, self.blurriness_tracker]

    def diffusion_schedule(self, diffusion_times: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        start_angle = K.cast(tf.math.acos(self.maximum_signal_rate), "float32")
        end_angle = K.cast(tf.math.acos(self.minimum_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = K.cos(diffusion_angles)
        noise_rates = K.sin(diffusion_angles)

        return noise_rates, signal_rates

    @LazyProperty
    def input_noise(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Input(shape=(1, 1, 1))

    @LazyProperty
    def input_layers(self) -> typing.Sequence[tf.keras.layers.Layer]:
        return self.input_layer, self.input_noise

    def set_encoded_layers(self) -> None:
        logger("Set encoder", indent=1)
        embedding = get_embedding(self.embedding_min_frequency, self.embedding_max_frequency, self.embedding_dims)
        embedded_noise = tf.keras.layers.Lambda(embedding, output_shape=(1, 1, self.embedding_dims))(self.input_noise)
        embedded_noise = tf.keras.layers.UpSampling2D(
            size=self.input_shape[:-1],
            interpolation="nearest"
        )(embedded_noise)
        current_layer = tf.keras.layers.Concatenate()([self.input_layer, embedded_noise])

        current_layer, self._encoded_inherited_layers = build_encoder(
            current_layer,
            self.layer_kernels,
        )
        self._encoded_layer = tf.keras.layers.Lambda(lambda x: K.tanh(x))(current_layer)
        logger("Set encoder OK", indent=-1)

    @LazyProperty
    def output_layer(self) -> tf.keras.layers.Layer:
        current_layer = build_decoder(
            self.encoded_layer,
            self.encoded_inherited_layers,
            self.layer_kernels,
        )
        output_layer = tf.keras.layers.Conv2D(
            self.output_shape[-1],
            activation="linear",
            kernel_size=1
        )(current_layer)
        return output_layer

    def denoise(
            self,
            noisy_images: tf.Tensor,
            noise_rates: tf.Tensor,
            signal_rates: tf.Tensor,
            training: bool
    ) -> typing.Tuple[tf.Tensor, tf.Tensor]:

        model = self.model if training else self.ema_model
        pred_images = model([noisy_images, noise_rates], training=training)
        pred_noises = (pred_images * signal_rates - noisy_images) / noise_rates
        return pred_noises, pred_images

    def reverse_diffusion(
            self,
            initial_noise: tf.Tensor,
            diffusion_steps: int,
            return_steps: bool = False
    ) -> typing.Union[np.ndarray, tf.Tensor]:
        step_size = 1.0 / diffusion_steps
        pred_images = None
        next_noisy_images = initial_noise
        steps = []

        for i_step, step in enumerate(range(diffusion_steps)):
            noisy_images = next_noisy_images
            diffusion_times = K.ones((self.batch_size, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
            steps.append(np.concatenate([noisy_images, pred_images], axis=2))

        steps = np.concatenate(steps, axis=0)
        if return_steps:
            return steps
        return pred_images

    def generate(self, diffusion_steps: int, return_steps: bool = False) -> tf.Tensor:
        initial_noise = K.random_normal(shape=(self.batch_size, *self.input_shape))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, return_steps=return_steps)
        return generated_images

    def step(
            self,
            images: tf.Tensor,
            noise_factor: float = noise_factor
    ) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        noises: tf.Tensor = K.random_normal(shape=(self.batch_size, *self.output_shape), stddev=noise_factor)
        diffusion_times = K.random_uniform(shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        return images, noises, noisy_images, noise_rates, signal_rates

    def call(self, images: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        images, noises, noisy_images, noise_rates, signal_rates = self.step(images)
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
        return pred_noises, pred_images

    def step_wrapper(
            self,
            packed_images: typing.Tuple[tf.Tensor, tf.Tensor],
            training: bool,
    ) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        image_inputs, image_outputs = packed_images
        images, noises, noisy_images, noise_rates, signal_rates = self.step(image_inputs)

        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=training
        )
        noise_loss = self.model.loss(noises, pred_noises)
        image_loss = self.model.loss(image_outputs, pred_images)
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        return images, image_loss

    def train_step(
            self,
            packed_images: typing.Tuple[Array_or_Tensor, Array_or_Tensor]
    ) -> typing.Dict[str, Array_or_Tensor]:
        self.training_mode = True

        with tf.GradientTape() as tape:
            _, image_loss = self.step_wrapper(packed_images, training=True)
        gradients = tape.gradient(image_loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        for weight, ema_weight in zip(self.model.weights, self.ema_model.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(
            self,
            packed_images: typing.Tuple[Array_or_Tensor, Array_or_Tensor]
    ) -> typing.Dict[str, Array_or_Tensor]:
        self.training_mode = False

        images, _ = self.step_wrapper(packed_images, training=False)
        generated_images = self.generate(diffusion_steps=self.kid_diffusion_steps)
        # self.kid.update_state(images, generated_images)
        self.blurriness_tracker.update_state(images, generated_images)
        # self.encoding_similarity.update_state(images, generated_images)
        return {m.name: m.result() for m in self.metrics}

    def fit(self,
            dataset: BatchGenerator,
            batch_size: int,
            validation_data: BatchGenerator,
            steps_per_epoch: int,
            validation_steps: int,
            epochs: int,
            callbacks: typing.Sequence[tf.keras.callbacks.Callback],
            class_weight: None = None
            ):
        if not class_weight:
            warnings.warn("`class_weight` not yet implemented. Will ignore the parameter.")

        self.on_fit_start(callbacks)

        logger.set_iterator(epochs)
        for epoch in range(epochs):
            logs = self.run_epoch(dataset, validation_data, steps_per_epoch, validation_steps)
            self.on_epoch_end(epoch, logs, callbacks)
            message = ', '.join([f"{key}: {value:.3f}" for key, value in logs.items()])
            logger.iterate(message)
        return

    def on_fit_start(self, callbacks: typing.Sequence[tf.keras.callbacks.Callback]) -> None:
        self.model.train_step = self.train_step
        self.model.test_step = self.test_step

    def run_epoch(
            self,
            dataset: BatchGenerator,
            validation_data: BatchGenerator,
            steps_per_epoch: int,
            validation_steps: int
    ) -> typing.Dict[str, Array_or_Tensor]:
        logs = {}
        logs.update(self.run_step(self.train_step, dataset, steps_per_epoch))
        logs.update(self.run_step(self.test_step, validation_data, validation_steps, key_prefix='val_'))
        return logs

    @staticmethod
    def run_step(
            step_function: typing.Callable[
                [typing.Tuple[Array_or_Tensor, Array_or_Tensor]],
                typing.Dict[str, Array_or_Tensor]
            ],
            dataset: BatchGenerator,
            num_steps: int,
            key_prefix: str = ""
    ) -> typing.Dict[str, Array_or_Tensor]:
        logs: typing.Dict[str, Array_or_Tensor] = {}
        for _ in range(num_steps):
            packed_images = next(iter(dataset))
            metrics = step_function(packed_images)
            for key, value in metrics.items():
                full_key = f"{key_prefix}{key}"
                logs[full_key] = logs.get(full_key, 0) + value.numpy() / num_steps
        return logs

    @staticmethod
    def on_epoch_end(epoch, logs, callbacks):
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=logs)
