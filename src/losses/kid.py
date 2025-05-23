import typing

import tensorflow as tf
import tensorflow.keras.backend as K


class KID(tf.keras.metrics.Metric):
    KID_IMAGE_SIZE: int = 75

    def get_default_encoder(self) -> tf.keras.models.Model:
        layers = [
            tf.keras.layers.experimental.preprocessing.Resizing(
                height=self.KID_IMAGE_SIZE,
                width=self.KID_IMAGE_SIZE
            ),
            tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
            tf.keras.applications.InceptionV3(
                include_top=False,
                input_shape=(self.KID_IMAGE_SIZE, self.KID_IMAGE_SIZE, 3),
                weights="imagenet",
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
        ]
        return tf.keras.models.Sequential(layers)

    def __init__(
            self,
            name,
            input_shape: typing.Sequence[int],
            layers: typing.Optional[typing.Sequence[tf.keras.layers.Layer]] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.kid_tracker = tf.keras.metrics.Mean(name="kid_tracker")

        input_layer = tf.keras.layers.Input(shape=input_shape)
        layers = (input_layer, *(self.get_default_encoder() if layers is None else layers))
        self.encoder = tf.keras.models.Sequential(layers)

    def polynomial_kernel(self, features_1: tf.Tensor, features_2: tf.Tensor) -> tf.Tensor:
        feature_dimensions = K.cast(K.shape(features_1)[1], dtype="float32")
        return (features_1 @ K.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(
            self,
            real_images: tf.Tensor,
            generated_images: tf.Tensor,
            sample_weight=None
    ) -> None:
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_size = real_features.shape[0]
        batch_size_f = K.cast(batch_size, dtype="float32")

        eye: tf.Tensor = (1.0 - K.eye(batch_size))
        norm = batch_size_f * (batch_size_f - 1.0)
        mean_kernel_real = K.sum(kernel_real * eye) / norm
        mean_kernel_generated = K.sum(kernel_generated * eye) / norm
        mean_kernel_cross = K.mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        self.kid_tracker.update_state(kid)

    def result(self) -> float:
        return self.kid_tracker.result()

    def reset_state(self) -> None:
        self.kid_tracker.reset_state()
