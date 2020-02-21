import sys
import os

from keras.models import Model
from keras.layers import Input, Conv2D
from keras_radam.training import RAdamOptimizer

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block, deconvolution_block
from Rignak_DeepLearning.config import get_config

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False
LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'heatmap'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def import_model(weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")
    convs = []
    block = None

    batch_normalization = config.get('BATCH_NORMALIZATION', False)
    conv_layers = config['CONV_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (512, 512, 3))
    output_shape = config.get('INPUT_SHAPE', (32, 32, 1))
    activation = config.get('ACTIVATION', 'relu')
    last_activation = config.get('ACTIVATION', 'sigmoid')
    loss = config.get('LOSS', 'mse')
    output_canals = output_shape

    inputs = Input(input_shape)
    # encoder
    for neurons in conv_layers:
        if block is None:
            block, conv = convolution_block(inputs, neurons, activation=activation, maxpool=True,
                                            batch_normalization=batch_normalization)
        else:
            block, conv = convolution_block(block, neurons, activation=activation, maxpool=True,
                                            batch_normalization=batch_normalization)
        convs.append(conv)

    # central
    block, conv = convolution_block(block, conv_layers[-1] * 2, activation=activation, maxpool=False,
                                    batch_normalization=batch_normalization)

    # decoder
    for neurons, previous_conv in zip(conv_layers[::-1], convs[::-1]):
        block = deconvolution_block(block, previous_conv, neurons, activation=activation)
        if block.output_shape[-3] == output_shape[0] and block.output_shape[-2] == output_shape[1]:
            conv_layer = Conv2D(output_canals, (1, 1), activation=last_activation)(block)
            break

    model = Model(inputs=[inputs], outputs=[conv_layer])
    optimizer = RAdamOptimizer(learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    model.name = name
    model.weight_filename = weight_filename
    if load:
        print('load weights')
        model.load_weights(weight_filename)

    with open(summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old

    return model