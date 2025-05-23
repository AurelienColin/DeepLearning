from src.generators.image_to_tag.custom.rating_as_float_generator import RatingAsFloatGenerator
from src.trainers.image_to_tag_trainers.run.rating_tagger import RatingTagger
from Rignak.lazy_property import LazyProperty
import typing
import tensorflow as tf
from src.callbacks.example_callback_with_logs import ExampleCallbackWithLogs
from src.callbacks.plotters.image_to_tag.regression_plotter import RegressionPlotter
from src.models.image_to_tag.regression_wrapper import RegressionWrapper
class RatingAsFloatTagger(RatingTagger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_generator = RatingAsFloatGenerator

    @property
    def get_model_wrapper(self) -> typing.Type:
        return RegressionWrapper

    @LazyProperty
    def callbacks(self) -> typing.Sequence[tf.keras.callbacks.Callback]:
        callbacks = super().callbacks[:-1]
        callbacks.append(
            ExampleCallbackWithLogs(
                model_wrapper=self.model_wrapper,
                output_path=self.model_wrapper.output_folder + "/regression",
                function=RegressionPlotter(
                    self.callback_generator,
                    self.validation_steps,
                    self.model_wrapper
                ),
                keep_all_epochs=False
            )
        )

        return callbacks




if __name__ == "__main__":
    # python src/trainers/image_to_tag_trainers/run/rating_as_float_tagger.py
    trainer = RatingAsFloatTagger()
    trainer.run()
