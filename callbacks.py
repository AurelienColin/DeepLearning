import os
import numpy as np
import matplotlib
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.callbacks import Callback

from Rignak_Misc.plt import COLORS
from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.Image_to_Image.plot_example import plot_example as plot_autoencoder_example
from Rignak_DeepLearning.Image_to_Class.plot_example import plot_example as plot_categorizer_example
from Rignak_DeepLearning.Image_to_Class.plot_example import plot_regressor

from Rignak_DeepLearning.Image_to_Class.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix

HISTORY_CALLBACK_ROOT = get_local_file(__file__, os.path.join('_outputs', 'history'))
EXAMPLE_CALLBACK_ROOT = get_local_file(__file__, os.path.join('_outputs', 'example'))
CONFUSION_CALLBACK_ROOT = get_local_file(__file__, os.path.join('_outputs', 'confusion'))


class HistoryCallback(Callback):
    """Callback generating a fitness plot in a file after each epoch"""

    def __init__(self, batch_size, training_steps, root=HISTORY_CALLBACK_ROOT):
        super().__init__()
        self.x = []
        self.logs = {}
        self.root = root
        self.batch_size = batch_size
        self.training_steps = training_steps

        self.metrics = []
        self.val_metrics = []

    def on_train_begin(self, logs=None):
        filename = os.path.join(self.root, f'{self.model.name}.png')
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        base_metrics = ('acc', 'val_acc', 'loss', 'val_loss')
        plt.ioff()
        self.x.append((epoch + 1) * self.batch_size * self.training_steps / 1000)

        for key, value in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(value)
        validation_logs = {key: value for key, value in self.logs.items()
                           if key not in base_metrics and key.startswith('val_')}
        cols = 1 if 'acc' not in logs and not len(validation_logs) else 2
        plt.figure(figsize=(6 * cols, 6))

        if 'acc' in logs:
            plt.subplot(1, cols, 2)
            plt.plot(self.x, self.logs['acc'], label="Training")
            plt.plot(self.x, self.logs['val_acc'], label="Validation")
            plt.xlabel('kimgs')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.legend()

        if len(validation_logs):
            plt.subplot(1, cols, 2)

            for color, (label, values) in zip(COLORS, validation_logs.items()):
                plt.plot(self.x, values, label=label, color=color)
            plt.xlabel('kimgs')
            plt.ylabel('Additional Validation Losses')
            plt.yscale('log')
            plt.legend()

        plt.subplot(1, cols, 1)

        plt.plot(self.x, self.logs['loss'], label="Training")
        plt.plot(self.x, self.logs['val_loss'], label="Validation")
        plt.xlabel('kimgs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.root, f'{self.model.name}.png'))
        plt.close()


class AutoencoderExampleCallback(Callback):
    def __init__(self, generator, root=EXAMPLE_CALLBACK_ROOT, denormalizer=None):
        super().__init__()
        self.root = root
        self.generator = generator
        self.denormalizer = denormalizer

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.join(self.root, self.model.name), exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        example = [[], [], []]
        while len(example[0]) < 8:
            next_ = next(self.generator)
            example[0] += list(next_[0])
            example[1] += list(next_[1])
            example[2] += list(self.model.predict(next_[0]))
        example[0] = np.array(example[0])
        example[1] = np.array(example[1])
        example[2] = np.array(example[2])

        plot_autoencoder_example(example[0], example[2], groundtruth=example[1], labels=self.model.callback_titles,
                                 denormalizer=self.denormalizer)
        plt.savefig(os.path.join(self.root, self.model.name, f'{os.path.split(self.model.name)[-1]}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


class ClassificationExampleCallback(Callback):
    def __init__(self, generator, root=EXAMPLE_CALLBACK_ROOT, denormalizer=None):
        super().__init__()
        self.root = root
        self.generator = generator
        self.denormalizer = denormalizer

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.join(self.root, self.model.name), exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        example = [[], [], []]
        while len(example[0]) < 8:
            next_ = next(self.generator)
            example[0] += list(next_[0])
            example[1] += list(next_[1])
            example[2] += list(self.model.predict(next_[0]))
        example[0] = np.array(example[0])
        example[1] = np.array(example[1])
        example[2] = np.array(example[2])

        plot_categorizer_example(example[:2], example[2], self.model.labels, denormalizer=self.denormalizer)
        plt.savefig(os.path.join(self.root, self.model.name, f'{os.path.split(self.model.name)[-1]}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


class RegressorCallback(Callback):
    def __init__(self, generator, validation_steps, attributes, means, stds,
                 root=EXAMPLE_CALLBACK_ROOT, denormalizer=None):
        super().__init__()
        self.root = root
        self.generator = generator
        self.denormalizer = denormalizer
        self.validation_steps = validation_steps
        self.means = means
        self.stds = stds
        self.attributes = attributes

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.join(self.root, self.model.name), exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        examples, truths, predictions = [], [], []
        if "".join(self.attributes) == 'RGB':
            while len(examples) < 12:
                next_ = next(self.generator)
                examples += list(next_[0])
                truths += list(next_[1])
                predictions += list(self.model.predict(next_[0]))
        else:
            for _ in range(self.validation_steps):
                next_ = next(self.generator)
                examples += list(next_[0])
                truths += list(next_[1])
                predictions += list(self.model.predict(next_[0]))

        plot_regressor(examples, np.array(truths), np.array(predictions), self.model.labels, self.means, self.stds)
        plt.savefig(os.path.join(self.root, self.model.name, f'{os.path.split(self.model.name)[-1]}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


class ConfusionCallback(Callback):
    def __init__(self, generator, labels, root=CONFUSION_CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.labels = labels

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.join(self.root, self.model.name), exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        confusion_matrix = compute_confusion_matrix(self.model, self.generator, canals=len(self.labels))
        plot_confusion_matrix(confusion_matrix, labels=self.labels)

        plt.savefig(os.path.join(self.root, self.model.name, f'{os.path.split(self.model.name)[-1]}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


class SaveAttributes(Callback):
    def __init__(self, generator, config, labels=None, max_examples=4):
        super().__init__()
        self.generator = generator
        self.config = config
        self.saved_logs = []
        self.labels = labels
        self.max_examples = max_examples

    def on_train_begin(self, logs=None):
        filename = self.model.weight_filename + '.json'
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.saved_logs.append(logs.copy())
        input_, groundtruth = next(self.generator)
        if len(input_) > self.max_examples:
            input_ = input_[:self.max_examples]
            groundtruth = groundtruth[:self.max_examples]

        val_losses = [log['val_loss'] for log in self.saved_logs]
        dict_to_save = {"_logs": self.saved_logs, "_labels": self.labels, "_config": self.config}
        with open(self.model.weight_filename + '.json', 'w') as file:
            json.dump(dict_to_save, file, sort_keys=True, indent=4)

        if np.argmax(val_losses) == len(val_losses) - 1:
            self.input_ = input_.tolist()
            self.groundtruth = groundtruth.tolist()
            self.output = self.model.predict(input_).tolist()

        samples_to_save = {'input': self.input_,
                           "output": self.output,
                           "groundtruth": self.groundtruth}
        with open(self.model.weight_filename + '_samples.json', 'w') as file:
            json.dump(samples_to_save, file, sort_keys=True, indent=4)
