import numpy as np
from rignak.src.custom_display import Display

from src.callbacks.plotters.plotter import Plotter, reset_display
from src.models.image_to_image.diffusion_model_wrapper import DiffusionModelWrapper
from src.callbacks.plotters.plot_from_arrays import PlotterFromArrays

class DiffusionExamplePlotter(PlotterFromArrays):
    model_wrapper: DiffusionModelWrapper

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, model_wrapper: DiffusionModelWrapper):
        ncols = min(inputs.shape[0], 4)
        inputs = inputs[:ncols]
        outputs = outputs[:4]

        thumbnail_size = super().thumbnail_size[0] * 4, super().thumbnail_size[1]
        super().__init__(inputs, outputs, model_wrapper,  ncols=ncols, nrows=8, thumbnail_size=thumbnail_size)

    @reset_display
    def __call__(self) -> Display:
        noises = np.random.normal(size=self.inputs.shape) * self.model_wrapper.noise_factor
        diffusion_times = np.arange(0, 1.001, 1 / (self.nrows - 1))
        diffusion_times = np.reshape(diffusion_times, (diffusion_times.shape[0], 1, 1, 1, 1))
        diffusion_times = np.repeat(diffusion_times, self.inputs.shape[0], axis=1)

        for i_row, diffusion_time in enumerate(diffusion_times):
            noise_rates, signal_rates = self.model_wrapper.diffusion_schedule(diffusion_time)
            noisy_images = signal_rates * self.inputs + noise_rates * noises

            pred_images = self.model_wrapper.model([noisy_images, noise_rates], training=False).numpy()[:, :, :, :3]
            error = np.abs(pred_images - self.inputs)

            images = self.concatenate(self.inputs, noisy_images, pred_images, error)
            for j_col, image in enumerate(images):
                self.imshow(i_row * self.ncols + j_col, image)
        return self.display
