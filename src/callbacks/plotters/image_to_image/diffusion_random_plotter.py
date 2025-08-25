import numpy as np
from rignak.src.custom_display import Display

from src.callbacks.plotters.plotter import Plotter, reset_display
from src.models.image_to_image.diffusion_model_wrapper import DiffusionModelWrapper
from src.models.model_wrapper import ModelWrapper
from src.callbacks.plotters.plot_from_arrays import PlotterFromArrays


class DiffusionRandomPlotter(PlotterFromArrays):
    model_wrapper: DiffusionModelWrapper

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, model_wrapper: ModelWrapper):
        thumbnail_size = super().thumbnail_size[0] * 2, super().thumbnail_size[1]
        super().__init__(
            inputs,
            outputs,
            model_wrapper,
            ncols=inputs.shape[0],
            nrows=5,
            thumbnail_size=thumbnail_size
        )

    @reset_display
    def __call__(self) -> Display:
        images_from_noise = self.model_wrapper.generate(self.nrows, return_steps=True)
        for i, image in enumerate(images_from_noise):
            self.imshow(i, image)
        return self.display
