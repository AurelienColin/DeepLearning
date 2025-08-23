import numpy as np
from rignak.custom_display import Display

from src.callbacks.plotters.plotter import reset_display
from src.models.model_wrapper import ModelWrapper
from src.callbacks.plotters.plot_from_arrays import PlotterFromArrays


class ImageToImageExamplePlotter(PlotterFromArrays):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, model_wrapper: ModelWrapper):
        ncols = 4
        nrows = inputs.shape[0]

        super().__init__(inputs, outputs, model_wrapper, ncols=ncols, nrows=nrows)

    @reset_display
    def __call__(self) -> Display:
        pred_images = self.model_wrapper.model(self.inputs, training=False).numpy()
        error = pred_images[:, :, :, :3] - self.outputs[:, :, :, :3]

        kwargs = [{}, {}, {}, {}]
        if self.outputs.shape[-1] == 1:
            kwargs[1] = kwargs[2] = dict(vmin=0, vmax=1, cmap_name="magma")
            kwargs[3] = dict(vmin=-1, vmax=1, cmap_name="Spectral")

        for i, (input, output, pred, error) in enumerate(zip(self.inputs, self.outputs, pred_images, error)):
            self.imshow(self.ncols*i, input, title="Input", **kwargs[0])
            self.imshow(self.ncols*i + 1, output, title="Truth", **kwargs[1])
            self.imshow(self.ncols*i + 2, pred, title="Pred.", **kwargs[2])
            self.imshow(self.ncols*i + 3, error, title="Err.", **kwargs[3])
        return self.display
