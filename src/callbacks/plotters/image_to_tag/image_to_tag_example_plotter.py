import typing

import numpy as np
from rignak.custom_display import Display
from rignak.lazy_property import LazyProperty

import src.trainers.image_to_image_trainers.run.benchmark.utils
from src.callbacks.plotters.plot_from_arrays import PlotterFromArrays
from src.callbacks.plotters.plotter import reset_display
from src.losses.losses import cross_entropy, one_minus_dice
from src.models.model_wrapper import ModelWrapper
from src.output_spaces.output_space import OutputSpace


class ImageToTagExamplePlotter(PlotterFromArrays):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, model_wrapper: ModelWrapper, output_space: OutputSpace):
        ncols = min(inputs.shape[0], 10)
        inputs = inputs[:ncols]
        outputs = outputs[:ncols]
        nrows = 3
        thumbnail_size = (6, 5)
        super().__init__(inputs, outputs, model_wrapper, ncols=ncols, nrows=nrows, thumbnail_size=thumbnail_size)
        self.max_tags: int = min(10, outputs.shape[1])
        self.logs: typing.Optional[np.ndarray] = None
        self._indices: typing.Optional[np.ndarray[int]] = None
        self.output_space: OutputSpace = output_space

    @LazyProperty
    def indices(self):
        indices = []
        for output in self.outputs:
            truth_indices = np.argwhere(output == 1)[:, 0]
            if len(truth_indices) < self.max_tags:
                remaining_indices = [i for i in range(self.output_space.n) if i not in truth_indices]
                other_indices = np.random.choice(
                    remaining_indices,
                    self.max_tags - truth_indices.shape[0],
                    replace=False,
                )
                stacked_indices = np.concatenate((truth_indices, other_indices))
                indices.append(stacked_indices)
            else:
                indices.append(truth_indices[:self.max_tags])
            indices[-1] = sorted(indices[-1])
        indices = np.array(indices)
        return indices

    def get_labels(self, indices: np.ndarray[int]) -> typing.Sequence[str]:
        return [self.output_space.tag_names[j] for j in indices]

    def call_for_inputs(self):
        for i, image in enumerate(self.inputs):
            self.imshow(i, image, interpolation="bicubic")

    def call_for_predictions(self):
        for i, (indices, output, prediction) in enumerate(zip(self.indices, self.outputs, self.logs[-1])):
            dice_value = 1 - one_minus_dice(output, prediction).numpy()
            crossentropy_value = cross_entropy(output, prediction).numpy()

            title = f"Dice: {dice_value:.0%}, CE: {crossentropy_value:.2f}"

            indices = indices[::-1]  # Reversing to have same order as call_for_logs
            reduced_output = output[indices]
            reduced_prediction = prediction[indices]

            labels = self.get_labels(indices)

            exponent = 3
            kwargs = dict(alpha=0.5, title=title, xscale="logit", epsilon=10 ** -exponent)
            for values, color in zip((reduced_output, reduced_prediction), ('tab:blue', 'tab:orange')):
                subplot = self.display[self.ncols + i]
                subplot.barh(labels, values, color=color, **kwargs)

                x_tick_labels = (
                    *[f"$10^{{{x}}}$" for x in range(-exponent, 0)],
                    "$\\frac{1}{2}$",
                    *[f"$1-10^{{{x}}}$" for x in range(1, exponent)]
                )
                subplot.ax.set_xticklabels(x_tick_labels)

    def call_for_logs(self):
        for i, indices in enumerate(self.indices):
            limit = 1e-3
            y = np.clip(self.logs[:, i, indices], limit, 1 - limit)
            self.display[i].plot(
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
