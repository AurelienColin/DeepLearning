from rignak.src.custom_display import Display
from rignak.src.lazy_property import LazyProperty

import numpy as np

from src.callbacks.plotters.plot_from_arrays import PlotterFromArrays
import typing


class ImageToTagExamplePlotter(PlotterFromArrays):
    def __init__(self, *args, **kwargs):
        self.max_tags = None
        self.output_space = None
        super().__init__(*args, **kwargs)

    @LazyProperty
    def indices(self):
        indices = []
        for output in self.outputs:
            truth_indices = np.argwhere(output == 1)[:, 0]
            if len(truth_indices) < self.max_tags:
                remaining_indices = [i for i in range(len(self.output_space)) if i not in truth_indices]
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

    def subcall_for_prediction(
            self,
            i: int,
            output: np.ndarray,
            prediction: np.ndarray,
            labels: typing.Sequence[str],
            title: str
    ) -> None:
        subplot = self.display[i]
        exponent = 3
        kwargs = dict(alpha=0.5, title=title, xscale="logit", epsilon=10 ** -exponent)
        for values, color in zip((output, prediction), ('tab:blue', 'tab:orange')):
            values = np.clip(values, kwargs['epsilon'], 1 - kwargs['epsilon'])
            subplot.barh(labels, values, color=color, **kwargs)

            x_tick_labels = (
                *[f"$10^{{{x}}}$" for x in range(-exponent, 0)],
                "$\\frac{1}{2}$",
                *[f"$1-10^{{{x}}}$" for x in range(1, exponent)]
            )
            subplot.ax.set_xticklabels(x_tick_labels)
            subplot.ax.set_ylim(-0.5, len(labels) - .5)

    def call_for_inputs(self, on: str = 'first_row'):
        for i, image in enumerate(self.inputs):
            if on == 'first_col':
                i = i * self.ncols
            self.imshow(i, image, interpolation="bicubic")
