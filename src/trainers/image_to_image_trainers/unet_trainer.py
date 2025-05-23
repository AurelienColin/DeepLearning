import typing

from src.generators.image_to_image.image_to_image_generator import ImageToImageGenerator
from src.models.image_to_image.unet_wrapper import UnetWrapper
from src.trainers.image_to_image_trainers.autoencoder_trainer import AutoEncoderTrainer


class UnetTrainer(AutoEncoderTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_generator = ImageToImageGenerator

    @property
    def get_model_wrapper(self) -> typing.Type:
        return UnetWrapper
