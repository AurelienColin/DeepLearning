import typing

import numpy as np

from src.callbacks.plotters.plotter import reset_display
from src.losses.losses import one_minus_dice, cross_entropy
from src.models.model_wrapper import ModelWrapper
from src.output_spaces.output_space import OutputSpace
from src.callbacks.plotters.image_to_tag.example_plotter.image_to_tag_example_plotter import ImageToTagExamplePlotter
from rignak.src.custom_display import Display
from rignak.src.lazy_property import LazyProperty

class ImageToFlatTagExamplePlotter(ImageToTagExamplePlotter):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, model_wrapper: ModelWrapper, output_space: OutputSpace):
        ncols = min(inputs.shape[0], 10)
        inputs = inputs[:ncols]
        outputs = outputs[:ncols]
        nrows = 3
        thumbnail_size = (6, 5)
        super().__init__(inputs, outputs, model_wrapper, ncols=ncols, nrows=nrows, thumbnail_size=thumbnail_size)
        self.max_tags: int = min(10, outputs.shape[1])
        self.logs: typing.Optional[np.ndarray] = None
        self._indices: typing.Optional[np.ndarray] = None
        self.output_space: OutputSpace = output_space


    def get_labels(self, indices: np.ndarray) -> typing.Sequence[str]:
        return [self.output_space.tag_names[j] for j in indices]




    def call_for_predictions(self)->None:
        for i, (indices, output, prediction) in enumerate(zip(self.indices, self.outputs, self.logs[-1])):
            dice_value = 1 - one_minus_dice(output, prediction).numpy()
            crossentropy_value = cross_entropy(output, prediction).numpy()

            title = f"Dice: {dice_value:.0%}, CE: {crossentropy_value:.2f}"

            indices = indices[::-1]  # Reversing to have same order as call_for_logs
            labels = self.get_labels(indices)
            reduced_output = output[indices]
            reduced_prediction = prediction[indices]
            self.subcall_for_prediction(2*self.ncols + i, reduced_output, reduced_prediction, labels, title)

    def call_for_logs(self):
        for i, indices in enumerate(self.indices):
            limit = 1e-3
            y = np.clip(self.logs[:, i, indices], limit, 1 - limit)
            self.display[self.ncols + i].plot(
                None,
                y,
                ylabel="Prediction",
                xlabel="Epochs",
                labels=self.get_labels(indices),
                xmin=0,
                ymin=limit / 2,
                ymax=1 - limit / 2,
                yscale="logit"
            )

    @reset_display
    def __call__(self) -> Display:
        preds = self.model_wrapper.model(self.inputs, training=False).numpy()
        if self.logs is None:
            self.logs = preds[np.newaxis]
        else:
            self.logs = np.concatenate((self.logs, preds[np.newaxis]))

        self.call_for_inputs()
        self.call_for_predictions()
        self.call_for_logs()

        return self.display
