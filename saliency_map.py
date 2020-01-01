import os
import sys

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import Rignak_DeepLearning.deprecation_warnings
from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_Misc.plt import imshow

from keras.models import load_model
import keras.backend as K

# MASK_SIZE = 64
# STRIDE = 8
DATASET = sys.argv[1]
MODEL_FILENAME = f"{DATASET}_inceptionV3_False.h5"


def get_process_image(model_filename=MODEL_FILENAME, mixed_id=3):
    def process_image(im):
        pooled_gradients_value, feature_layer_value = get_gradients([np.expand_dims(im, axis=0)])
        for i in range(class_number):
            feature_layer_value[:, :, i] *= pooled_gradients_value[i]

        heatmap = np.mean(feature_layer_value, axis=-1)
        heatmap = np.maximum(heatmap, 0.0)
        heatmap = np.minimum(heatmap, 1.2)
        heatmap = np.square(heatmap)
        heatmap = (heatmap - np.min(heatmap)) / 1.2**2
        return heatmap

    model_filename = os.path.join('_outputs', 'models', model_filename)
    model = load_model(model_filename)
    class_number = model.get_layer('dense_1').output_shape[-1]
    feature_layer = model.get_layer(f'mixed{mixed_id}')
    class_outputs = model.output
    gradient = K.gradients(class_outputs, feature_layer.output)[0]
    pooled_gradients = K.mean(gradient, axis=(0, 1, 2))
    get_gradients = K.function([model.input], [pooled_gradients, feature_layer.output[0]])
    return process_image


def plot(k, filename, im, heatmap, dataset=DATASET):
    plt.figure(figsize=(10, 6))

    plt.subplot(121)
    imshow(im)
    plt.title('Input')
    plt.subplot(122)
    imshow(heatmap)
    plt.title("Heatmap")
    plt.tight_layout()

    path, filename = os.path.split(filename)
    label = os.path.split(path)[-1]
    plt.suptitle(f"{label}: {filename} ")
    os.makedirs(os.path.join('_outputs', 'saliency_map', f"{dataset}_mixed{k}"), exist_ok=True)
    os.makedirs(os.path.join('_outputs', 'saliency_map', f"{dataset}_mixed{k}", label), exist_ok=True)
    plt.savefig(os.path.join('_outputs', 'saliency_map', f"{dataset}_mixed{k}", label, filename))


train_folder, val_folder = get_dataset_roots("inceptionV3", dataset=DATASET)
labels = [folder for folder in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, folder))]
all_filenames = {label: os.listdir(os.path.join(train_folder, label)) for label in labels}

filenames = {label: os.listdir(os.path.join(train_folder, label)) for label in labels}

for k in tqdm([3]):
    process_image = get_process_image(model_filename=MODEL_FILENAME, mixed_id=k)
    for j in trange(10):
        for i, label in enumerate(tqdm(labels)):
            filename = os.path.join(train_folder, label, filenames[label][j])

            im = np.array(PIL.Image.open(filename)) / 255
            if len(im.shape) == 2:
                im = np.expand_dims(im, -1)
            heatmap = process_image(im)
            plot(k, filename, im, heatmap)