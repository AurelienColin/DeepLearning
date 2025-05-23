from abc import ABC

from src.callbacks.plotters.plotter import Plotter
from src.generators.image_to_tag.classification_generator import ClassificationGenerator


class PlotterFromGenerator(Plotter, ABC):
    def __init__(self, generator: ClassificationGenerator, steps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator: ClassificationGenerator = generator
        self.steps: int = steps
