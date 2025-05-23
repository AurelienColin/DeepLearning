import typing
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class Loss:
    losses: typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor, typing.Optional[tf.Tensor], float], tf.Tensor]]
    class_weights: typing.Optional[typing.Union[tf.Tensor, typing.Sequence[float]]] = None
    loss_weights: typing.Optional[typing.Sequence] = None
    epsilon: float = 1e-7

    def __post_init__(self):
        loss_weights = np.ones(len(self.losses)) if self.loss_weights is None else self.loss_weights
        self.loss_weights = tf.cast(loss_weights, tf.float32)

        if self.class_weights is not None:
            class_weights = tf.convert_to_tensor(self.class_weights)
            self.class_weights = tf.cast(class_weights, tf.float32)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        total: typing.Optional[tf.Tensor] = None
        for weight, loss in zip(self.loss_weights, self.losses):
            current_loss: tf.Tensor = loss(y_true, y_pred, self.class_weights, self.epsilon)
            current_loss = current_loss * weight
            if total is None:
                total = current_loss
            else:
                total = total + current_loss
        return total

    @staticmethod
    def apply_class_weights(loss: tf.Tensor, class_weights: typing.Optional[tf.Tensor]) -> tf.Tensor:
        if class_weights is not None:
            axis = list(range(tf.rank(loss) - 1))
            loss = tf.reduce_mean(loss, axis=axis)
            loss *= class_weights
        return loss


def fourth_channel_mae(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        class_weights: None = None,
        epsilon: float = Loss.epsilon
) -> tf.Tensor:
    loss = mae(y_true[:, :, :, 3], y_pred[:, :, :, 3], None, epsilon)
    return tf.reduce_mean(loss)


def mae(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        class_weights: typing.Optional[tf.Tensor] = None,
        *args
) -> tf.Tensor:
    loss = tf.abs(y_true - y_pred)
    loss = Loss.apply_class_weights(loss, class_weights)
    return tf.reduce_mean(loss)


def cross_entropy(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        class_weights: typing.Optional[tf.Tensor] = None,
        epsilon: float = Loss.epsilon
) -> tf.Tensor:
    positive_loss = y_true * tf.math.log(y_pred + epsilon)
    negative_loss = (1 - y_true) * tf.math.log((1 - y_pred) + epsilon)
    loss = positive_loss + negative_loss
    loss = Loss.apply_class_weights(loss, class_weights)
    return -tf.reduce_mean(loss)


def one_minus_dice(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        class_weights: typing.Optional[tf.Tensor] = None,
        epsilon: float = Loss.epsilon
) -> tf.Tensor:
    numerator = epsilon + 2 * Loss.apply_class_weights(y_true * y_pred, class_weights)
    denominator = epsilon + Loss.apply_class_weights(y_true + y_pred, class_weights)
    dice = numerator / denominator
    return 1 - tf.reduce_mean(dice)


def std_difference(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        *args
) -> tf.Tensor:
    std_true = tf.math.reduce_std(y_true, axis=0)
    std_pred = tf.math.reduce_std(y_pred, axis=0)
    loss = tf.abs(std_true - std_pred)
    return tf.reduce_mean(loss)


def edge_loss(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        *args,
        epsilon: float = Loss.epsilon
) -> tf.Tensor:
    def sobel_edges(image: tf.Tensor) -> tf.Tensor:
        sobel_x = tf.image.sobel_edges(image)
        grad_x = sobel_x[..., 0]
        grad_y = sobel_x[..., 1]
        edges = tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + epsilon)
        return edges

    edges_true = sobel_edges(y_true)
    edges_pred = sobel_edges(y_pred)
    edge_loss_value = mae(edges_true, edges_pred, *args)

    return edge_loss_value
