import sys

import numpy as np
import typing
from src.callbacks.plotters.plotter import reset_display
from src.models.model_wrapper import ModelWrapper
from src.callbacks.plotters.image_to_tag.example_plotter.image_to_tag_example_plotter import ImageToTagExamplePlotter
from rignak.src.custom_display import Display
from rignak.src.lazy_property import LazyProperty
from src.output_spaces.comparator_from_filesystem import ComparatorSpace


class ComparatorExamplePlotter(ImageToTagExamplePlotter):
    imshow_kwargs: typing.Dict = dict(interpolation="nearest")

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, model_wrapper: ModelWrapper,
                 output_space: ComparatorSpace):
        ncols = min(inputs.shape[0], 10)
        inputs = inputs[:ncols]
        outputs = [output[:ncols] for output in outputs]
        nrows = 5
        thumbnail_size = (6, 5)
        super().__init__(
            inputs,
            outputs[0],
            model_wrapper,
            ncols=ncols, nrows=nrows, thumbnail_size=thumbnail_size
        )
        self.output_space: ComparatorSpace = output_space

    def call_for_inputs(self):
        for i, (first_image, second_image) in enumerate(self.inputs):
            self.imshow(i, first_image, **self.imshow_kwargs)
            self.imshow(self.ncols + i, second_image, **self.imshow_kwargs)

    def call_for_predictions(self) -> None:
        def get_index(batch_index: int, row_index: int) -> int:
            return batch_index + self.ncols * (row_index - 1)

        similarities, reconstructions_a, reconstructions_b = self.model_wrapper.model(self.inputs, training=False)
        similarities = similarities.numpy()
        reconstructions_a = reconstructions_a.numpy()
        reconstructions_b = reconstructions_b.numpy()

        labels = [f'Level {i}' for i in range(self.output_space.level)]
        for i, (output, similarity, reconstruction_a, reconstruction_b) in enumerate(zip(
                self.outputs, similarities, reconstructions_a, reconstructions_b
        )):
            self.subcall_for_prediction(get_index(i, 5), output, similarity, labels, "")
            self.imshow(get_index(i, 3), reconstruction_a, **self.imshow_kwargs)
            self.imshow(get_index(i, 4), reconstruction_b, **self.imshow_kwargs)

    @reset_display
    def __call__(self) -> Display:
        self.call_for_inputs()
        self.call_for_predictions()

        return self.display
