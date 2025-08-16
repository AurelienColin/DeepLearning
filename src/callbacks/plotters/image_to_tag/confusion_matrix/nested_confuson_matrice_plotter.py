import typing

import numpy as np
from rignak.custom_display import Display

from src.callbacks.plotters.plotter import reset_display
from src.callbacks.plotters.plotter_from_generator import PlotterFromGenerator
from src.callbacks.plotters.image_to_tag.confusion_matrix.confuson_matrice_plotter import ConfusionMatricePlotter


class NestedConfusionMatricePlotter(ConfusionMatricePlotter):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.categories = self.generator.output_space.categories
        self.nrows = 4
        self.ncols = int(len(self.categories) / self.nrows)

    def get_confusion_matrix(self):
        confusion_matrices = [
            np.zeros((len(category), len(category)))
            for category in self.categories
        ]

        for _ in range(self.steps):
            inputs, outputs = next(self.generator)
            outputs = outputs.astype(int)

            predictions = self.model_wrapper.model(inputs, training=False).numpy()

            jmin = 0
            for i, confusion_matrice in enumerate(confusion_matrices):
                jmax = jmin + len(self.categories[i])

                confusion_matrices[i] = self.update_confusion_matrice(
                    confusion_matrice,
                    outputs[:, jmin:jmax],
                    predictions[:, jmin:jmax]
                )
                jmin = jmax

        for i, confusion_matrice in enumerate(confusion_matrices):
            confusion_matrices[i] = confusion_matrice / np.sum(confusion_matrice, axis=1, keepdims=True)
        return confusion_matrices, {}

    @reset_display
    def __call__(self) -> typing.Tuple[Display, typing.Dict[str, typing.Sequence]]:
        confusion_matrices, logs = self.get_confusion_matrix()

        for i, (category, confusion_matrice) in enumerate(zip(self.categories, confusion_matrices)):
            accuracy = np.diagonal(confusion_matrice).mean()
            title = f"{category.name} | Acc.: {accuracy:.1%}"
            self.display[i].heatmap(
                confusion_matrice,
                ylabel="True labels",
                xlabel="Predicted labels",
                labels=category.labels,
                cmap_name="Blues",
                xticks_rotation=30,
                title=title,
                vmin=0, vmax=1
            )

        return self.display, logs
