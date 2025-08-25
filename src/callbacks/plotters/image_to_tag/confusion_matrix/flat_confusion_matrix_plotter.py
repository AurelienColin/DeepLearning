import typing

import numpy as np
from rignak.src.custom_display import Display

from src.callbacks.plotters.plotter import reset_display
from src.callbacks.plotters.plotter_from_generator import PlotterFromGenerator
from src.callbacks.plotters.image_to_tag.confusion_matrix.confuson_matrice_plotter import ConfusionMatricePlotter


class FlatConfusionMatricePlotter(ConfusionMatricePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @reset_display
    def __call__(self) -> typing.Tuple[Display, typing.Dict[str, typing.Sequence]]:
        confusion_matrice, logs = self.get_confusion_matrix()
        self.display[0].heatmap(
            confusion_matrice,
            ylabel="True labels",
            xlabel="Predicted labels",
            labels=self.generator.output_space.tag_names,
            cmap_name="Blues",
            xticks_rotation=30,
            vmin=0, vmax=1
        )
        return self.display, logs

    def get_categorization_report(self, results: np.ndarray) -> typing.Dict[str, np.ndarray]:
        true_positives = results[:, 2]
        true_negatives = results[:, 3]
        false_negatives = results[:, 4]
        false_positives = results[:, 5]

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        accuracy = (true_positives + true_negatives) / np.sum(results, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert len(self.generator.output_space.tag_names) == results.shape[0]

        return dict(
            Class=self.generator.output_space.tag_names,
            Truth=results[:, 0],
            Prediction=results[:, 1],
            Precision=precision,
            Recall=recall,
            Accuracy=accuracy,
            F1=f1_score
        )
