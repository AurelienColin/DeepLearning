"""
Mostly generated with gemini
"""

import typing

import tensorflow as tf
import numpy as np
import math
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger


class SparseConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 n_kernels: int,
                 kernel_size: int,
                 n_non_zero: int,
                 strides: int = 1,
                 padding: str = 'same',
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 n_stride: int = 1,
                 activation: typing.Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        n_non_zero = np.clip(n_non_zero, 1, kernel_size ** 2)

        # Store arguments
        self._original_kernels: int = n_kernels
        self._kernels: typing.Optional[int] = None
        self.kernel_size: typing.Tuple[int, int] = (kernel_size, kernel_size)
        self.n_non_zero: int = n_non_zero
        self.strides: typing.Tuple[int, int] = (strides, strides)
        self.padding: str = padding.upper()
        self.use_bias: bool = use_bias
        self.kernel_initializer: tf.keras.initializers.Initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer: tf.keras.initializers = tf.keras.initializers.get(bias_initializer)
        self.activation = tf.keras.activations.get(activation)

        self._indices: typing.Optional[tf.Tensor] = None
        self.sparse_kernel: typing.Optional[tf.Tensor] = None
        self.bias: typing.Optional[tf.Tensor] = None

    @LazyProperty
    def kernels(self) -> int:
        n_positions = self.kernel_size[0] * self.kernel_size[1]

        num = math.factorial(n_positions)
        denum = math.factorial(self.n_non_zero) * math.factorial(n_positions - self.n_non_zero)
        max_possible_kernels = int(num / denum)

        if self._original_kernels > max_possible_kernels:
            logger.warning(
                f"Requested {self._original_kernels} kernels. Capping kernels to {max_possible_kernels}."
            )
        return min(max_possible_kernels, self._original_kernels)

    @LazyProperty
    def indices(self) -> tf.Tensor:
        y, x = np.mgrid[0:self.kernel_size[0], 0:self.kernel_size[1]]
        all_coords = np.stack([y.ravel(), x.ravel()], axis=-1)
        n_total_coords = len(all_coords)

        chosen_patterns_set = set()
        # Loop until we have collected enough unique patterns
        while len(chosen_patterns_set) < self.kernels:
            rand_indices = np.random.choice(n_total_coords, size=self.n_non_zero, replace=False)
            combination = all_coords[rand_indices]
            combination_sorted = combination[np.lexsort((combination[:, 1], combination[:, 0]))]
            chosen_patterns_set.add(tuple(map(tuple, combination_sorted)))

        chosen_patterns = np.array(list(chosen_patterns_set))
        return tf.Variable(initial_value=chosen_patterns, dtype=tf.int32, trainable=False)

    def build(self, input_shape: typing.Sequence[int]):
        if self.built:
            return

        in_channels = int(input_shape[-1])

        self.sparse_kernel = self.add_weight(
            name='sparse_kernel',
            shape=(self.n_non_zero, in_channels, self.kernels),
            initializer=self.kernel_initializer,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.kernels,),
                initializer=self.bias_initializer,
                trainable=True
            )

        self.built = True

    def call(self, inputs):
        """
        Performs the forward pass of the layer.
        """
        in_channels = tf.shape(inputs)[-1]
        kernel_h, kernel_w = self.kernel_size

        # --- 1. Construct the full, dense kernel from the sparse representation ---
        # We use `tf.scatter_nd` to build the dense kernel on-the-fly.

        # First, prepare the indices for `scatter_nd`. We need to generate a
        # 4D index (h, w, in_channel, out_channel) for every non-zero weight.

        # Get the (h, w) coordinates for each filter's pattern
        # Shape: (kernels, n_non_zero, 2)
        hw_coords = self.indices

        # Tile these coordinates to prepare for merging with in/out channel indices
        # Shape becomes: (kernels, in_channels, n_non_zero, 2)
        hw_coords = tf.expand_dims(hw_coords, axis=1)
        hw_coords = tf.tile(hw_coords, [1, in_channels, 1, 1])

        # Create indices for the output channels (kernels)
        # Shape becomes: (kernels, in_channels, n_non_zero, 1)
        out_channel_indices = tf.range(self.kernels, dtype=tf.int32)
        out_channel_indices = tf.reshape(out_channel_indices, (self.kernels, 1, 1, 1))
        out_channel_indices = tf.tile(out_channel_indices, [1, in_channels, self.n_non_zero, 1])

        # Create indices for the input channels
        # Shape becomes: (kernels, in_channels, n_non_zero, 1)
        in_channel_indices = tf.range(in_channels, dtype=tf.int32)
        in_channel_indices = tf.reshape(in_channel_indices, (1, in_channels, 1, 1))
        in_channel_indices = tf.tile(in_channel_indices, [self.kernels, 1, self.n_non_zero, 1])

        # Concatenate all parts to form the final indices for scatter_nd
        # The final shape is (kernels, in_channels, n_non_zero, 4)
        scatter_indices = tf.concat([
            hw_coords[..., 0:1],  # h
            hw_coords[..., 1:2],  # w
            in_channel_indices,  # in_channel
            out_channel_indices  # out_channel
        ], axis=-1)

        # Now, prepare the `updates` tensor (the actual weight values)
        # We need to transpose `self.sparse_kernel` to match the order of `scatter_indices`
        # Original shape: (n_non_zero, in_channels, kernels)
        # Target shape: (kernels, in_channels, n_non_zero)
        updates = tf.transpose(self.sparse_kernel, perm=[2, 1, 0])

        # Construct the dense kernel
        dense_kernel_shape = (kernel_h, kernel_w, in_channels, self.kernels)
        dense_kernel = tf.scatter_nd(scatter_indices, updates, dense_kernel_shape)

        # --- 2. Perform the convolution using the reconstructed dense kernel ---
        outputs = tf.nn.conv2d(
            inputs,
            dense_kernel,
            strides=self.strides,
            padding=self.padding
        )

        # --- 3. Add bias if enabled ---
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        """Enables serialization of the layer."""
        config = super(SparseConv2D, self).get_config()
        config.update({
            'kernels': self.kernels,
            'kernel_size': self.kernel_size,
            'n_non_zero': self.n_non_zero,
            'strides': self.strides,
            'padding': self.padding.lower(),  # Save as lowercase string
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
        })
        return config
