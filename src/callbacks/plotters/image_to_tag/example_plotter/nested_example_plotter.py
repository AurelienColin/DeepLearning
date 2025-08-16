import typing

import numpy as np

from callbacks.plotters.plotter import reset_display
from losses.losses import one_minus_dice
from models.model_wrapper import ModelWrapper
from callbacks.plotters.image_to_tag.example_plotter.image_to_tag_example_plotter import ImageToTagExamplePlotter
from rignak.custom_display import Display
from rignak.lazy_property import LazyProperty
from output_spaces.custom.nested.nested_space import NestedSpace
import sys

class ImageToNestedTagExamplePlotter(ImageToTagExamplePlotter):
    def __init__(
            self,
            inputs: np.ndarray,
            outputs: np.ndarray,
            model_wrapper: ModelWrapper,
            output_space: NestedSpace
    ):
        nrows = min(inputs.shape[0], 8)

        inputs = inputs[:nrows]
        outputs = outputs[:nrows]
        ncols = 1 + len(output_space.categories)
        thumbnail_size = (6, 5)
        super().__init__(inputs, outputs, model_wrapper, ncols=ncols, nrows=nrows, thumbnail_size=thumbnail_size)
        self.output_space: NestedSpace = output_space

    # def get_labels(self, indices: np.ndarray[int]) -> typing.Sequence[str]:
    #     return [self.output_space.tag_names[j] for j in indices]

    def get_labels(self, indices: np.ndarray[int]) -> typing.Sequence[str]:
        return [self.output_space.categories[i].name for i in indices]

    def call_for_predictions(self) -> None:
        preds = self.model_wrapper.model(self.inputs, training=False).numpy()

        for i, (output, prediction) in enumerate(zip(self.outputs, preds)):
            jmin = 0
            for j, category in enumerate(self.output_space.categories):
                jmax = jmin + len(category)
                reduced_output = output[jmin:jmax]
                reduced_prediction = prediction[jmin:jmax]

                jmin = jmax
                labels = category.labels

                dice_value = 1 - one_minus_dice(reduced_output, reduced_prediction).numpy()

                title = f"{category.name} - Dice: {dice_value:.0%}"

                self.subcall_for_prediction(i * self.ncols + j + 1, reduced_output, reduced_prediction, labels, title)

    @reset_display
    def __call__(self) -> Display:
        self.call_for_inputs(on='first_col')
        self.call_for_predictions()

        return self.display
