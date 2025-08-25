import shutil
import typing

from rignak.src.custom_display import Display

from src.callbacks.callback import Callback


class ExampleCallback(Callback):

    def __init__(
            self,
            *args,
            function: typing.Callable[[], Display],
            period: int = 1,
            keep_all_epochs: bool = True,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.function: typing.Callable[[], Display] = function
        self.period: int = period
        self.keep_all_epochs: bool = keep_all_epochs

    def on_train_begin(self, logs: typing.Optional[typing.Dict[str, float]] = None) -> None:
        super().on_train_begin(logs=logs)
        self.on_epoch_end(0, logs=logs)

    def save_display(self, display: Display, epoch: int) -> typing.Sequence[str]:
        if self.keep_all_epochs:
            export_filenames = self.output_path + f"/{epoch:04d}.png", self.output_path + ".png"
        else:
            export_filenames = self.output_path + ".png",

        display.show(export_filename=export_filenames[0])
        for filename in export_filenames[1:]:
            shutil.copyfile(export_filenames[0], filename)
        return export_filenames

    def on_epoch_end(
            self,
            epoch: int,
            logs: typing.Optional[typing.Dict[str, float]] = None,
    ) -> None:
        if epoch % self.period:
            return

        display = self.function()
        self.save_display(display, epoch)
