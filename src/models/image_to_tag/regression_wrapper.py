import typing
from dataclasses import dataclass

import tensorflow as tf
from rignak.lazy_property import LazyProperty

from src.losses import losses
from src.models.image_to_tag.categorizer_wrapper import CategorizerWrapper


@dataclass
class RegressionWrapper(CategorizerWrapper):
    @LazyProperty
    def loss(self) -> typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        return losses.Loss((losses.mae,))

    @LazyProperty
    def metrics(self) -> typing.Sequence[typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        return ()
