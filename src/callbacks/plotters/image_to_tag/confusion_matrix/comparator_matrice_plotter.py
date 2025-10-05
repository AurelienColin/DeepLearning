import typing

import numpy as np
from rignak.src.custom_display import Display

from src.callbacks.plotters.plotter import reset_display
from src.callbacks.plotters.image_to_tag.confusion_matrix.confuson_matrice_plotter import ConfusionMatricePlotter


class ComparatorConfusionMatricePlotter(ConfusionMatricePlotter):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.ncols = self.level = self.generator.output_space.level
        self.nrows = 1

    def get_confusion_matrix(self):
        confusion_matrices = np.zeros((2, 2, self.level))

        for _ in range(self.steps):
            inputs, outputs = next(self.generator)
            outputs = outputs[0].astype(int)

            predictions = self.model_wrapper.model(inputs, training=False)[0].numpy()
            predictions = (predictions > 0.5).astype(int)

            true_positives = predictions * outputs
            true_negatives = (1 - predictions) * outputs
            false_negatives = predictions * outputs
            false_positives = predictions * (1 - outputs)

            for level in range(self.level):
                confusion_matrices[0, 0, level] = true_positives[:, level].sum()
                confusion_matrices[0, 1, level] = true_negatives[:, level].sum()
                confusion_matrices[1, 0, level] = false_positives[:, level].sum()
                confusion_matrices[1, 1, level] = false_negatives[:, level].sum()

        confusion_matrices = confusion_matrices / np.sum(confusion_matrices, axis=1, keepdims=True)
        return confusion_matrices, {}

    @reset_display
    def __call__(self) -> typing.Tuple[Display, typing.Dict[str, typing.Sequence]]:
        confusion_matrices, logs = self.get_confusion_matrix()

        for i in range(self.level):
            confusion_matrice=confusion_matrices[:,:,i]
            accuracy = np.diagonal(confusion_matrice).mean()
            title = f"Level {i} | Acc.: {accuracy:.1%}"
            self.display[i].heatmap(
                confusion_matrice,
                ylabel="True labels",
                xlabel="Predicted labels",
                labels=('Same', 'Different'),
                cmap_name="Blues",
                xticks_rotation=30,
                title=title,
                vmin=0, vmax=1
            )

        return self.display, logs
