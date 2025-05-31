import numpy as np
from rignak.custom_display import Display

from src.callbacks.plotters.plotter import Plotter, reset_display
from src.models.model_wrapper import ModelWrapper
from src.callbacks.plotters.plot_from_arrays import PlotterFromArrays


class ImageToImageExamplePlotter(PlotterFromArrays):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, model_wrapper: ModelWrapper):
        ncols = 4
        nrows = int(np.ceil(inputs.shape[0] / ncols))

        thumbnail_size = super().thumbnail_size[0] * 4, super().thumbnail_size[1]
        super().__init__(inputs, outputs, model_wrapper, ncols=ncols, nrows=nrows, thumbnail_size=thumbnail_size)

    @reset_display
    def __call__(self) -> Display:
        pred_images = self.model_wrapper.model(self.inputs, training=False).numpy()
        error = np.abs(pred_images[:, :, :, :3] - self.outputs[:, :, :, :3])
        images = self.concatenate(self.inputs, self.outputs, pred_images, error)

        for i, image in enumerate(images):
            self.imshow(i, image)
        return self.display
