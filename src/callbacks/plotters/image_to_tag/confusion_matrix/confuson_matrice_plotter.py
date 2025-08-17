import typing

import numpy as np
from rignak.custom_display import Display

from src.callbacks.plotters.plotter import reset_display
from src.callbacks.plotters.plotter_from_generator import PlotterFromGenerator


class ConfusionMatricePlotter(PlotterFromGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ncols=1, nrows=1, thumbnail_size=(10, 10))


    def update_confusion_matrice(
            self,
            _confusion_matrice: np.ndarray,
            _truth: np.ndarray,
            _prediction: np.ndarray
    ) -> np.ndarray:
        true_indices = np.argmax(_truth, axis=-1)
        pred_indices = np.argmax(_prediction, axis=-1)
        np.add.at(_confusion_matrice, (true_indices, pred_indices), 1)
        return _confusion_matrice

    def get_confusion_matrix(self):
        n = len(self.generator.output_space)
        results = np.zeros((n, 6), int)
        confusion_matrice = np.zeros((n, n))

        for _ in range(self.steps):
            inputs, outputs = next(self.generator)
            outputs = outputs.astype(int)

            predictions = self.model_wrapper.model(inputs, training=False).numpy()
            confusion_matrice = self.update_confusion_matrice(confusion_matrice, outputs, predictions)

            predictions = np.where(predictions > 0.5, 1, 0)
            results[:, 0] += np.sum(outputs, axis=0)  # True
            results[:, 1] += np.sum(predictions, axis=0)  # Prediction
            results[:, 2] += np.sum(predictions * outputs, axis=0)  # True Positives
            results[:, 3] += np.sum((1 - predictions) * (1 - outputs), axis=0)  # True Negatives
            results[:, 4] += np.sum(predictions * (1 - outputs), axis=0)  # False Positives
            results[:, 5] += np.sum((1 - predictions) * outputs, axis=0)  # False Negatives

        logs = self.get_categorization_report(results)

        confusion_matrice = confusion_matrice / np.sum(confusion_matrice, axis=1, keepdims=True)
        return confusion_matrice, logs