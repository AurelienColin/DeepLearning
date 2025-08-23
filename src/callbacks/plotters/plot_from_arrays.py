from abc import ABC

import numpy as np

from src.callbacks.plotters.plotter import Plotter


class PlotterFromArrays(Plotter, ABC):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs: np.ndarray = inputs
        self.outputs: np.ndarray = outputs

        if self.outputs.ndim == 3:
            self.outputs = np.expand_dims(self.outputs, -1)
