import typing

import numpy as np
from rignak.custom_display import Display
from scipy.stats import pearsonr

from src.callbacks.plotters.plotter import reset_display
from src.callbacks.plotters.plotter_from_generator import PlotterFromGenerator


class RegressionPlotter(PlotterFromGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ncols=1, nrows=1, thumbnail_size=(10, 10))
        self.ncols = self.generator.output_space.n

    @reset_display
    def __call__(self) -> typing.Tuple[Display, typing.Dict[str, typing.Sequence]]:
        n = self.generator.output_space.n

        results = np.zeros((self.generator.batch_size * self.steps, n, 2))
        for i in range(self.steps):
            k0 = i * self.generator.batch_size
            k1 = k0 + self.generator.batch_size

            inputs, outputs = next(self.generator)
            predictions = self.model_wrapper.model(inputs, training=False).numpy()

            results[k0:k1, :, 0] = outputs
            results[k0:k1, :, 1] = predictions

        logs = dict(tag=[], mae=[], pcc=[])
        for i in range(n):
            truth = results[:, i, 0]
            preds = results[:, i, 1]

            vmin = np.min(truth)
            vmax = np.max(truth)
            self.display[i].plot_regression(truth, preds, ylabel="Predicted value", xlabel="True value", xmin=vmin,
                                            ymin=vmin, xmax=vmax, ymax=vmax
                                            )
            logs['tag'].append(self.generator.output_space.tag_names[i])
            logs['mae'].append(np.mean(np.abs(truth - preds)))
            logs['pcc'].append(pearsonr(truth, preds).statistic)

        return self.display, logs
