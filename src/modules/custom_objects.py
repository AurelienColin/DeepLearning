import tensorflow as tf

from src.modules.blocks.convolution_block import ConvolutionBlock
from src.modules.blocks.deconvolution_block import DeconvolutionBlock
from src.modules.blocks.residual_block import ResidualBlock
from src.modules.layers.atrous_conv2d import AtrousConv2D
from src.modules.layers.padded_conv2d import PaddedConv2D
from src.modules.layers.scale_layer import ScaleLayer

CUSTOM_OBJECTS = dict(
    K=tf.keras.backend,
    ResidualBlock=ResidualBlock,
    DeconvolutionBlock=DeconvolutionBlock,
    ConvolutionBlock=ConvolutionBlock,
    ScaleLayer=ScaleLayer,
    AtrousConv2D=AtrousConv2D,
    PaddedConv2D=PaddedConv2D
)
