import os.path
import typing

import pandas as pd
from Rignak.custom_display import Display

from src.callbacks.example_callback import ExampleCallback


class ExampleCallbackWithLogs(ExampleCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function: typing.Callable[[], typing.Tuple[Display, typing.Iterable]]

    def on_epoch_end(
            self,
            epoch: int,
            logs: typing.Optional[typing.Dict[str, float]] = None,
    ) -> None:
        if epoch % self.period:
            return

        display, logs = self.function()
        export_filenames = self.save_display(display, epoch)
        pd.DataFrame(logs).to_csv(os.path.splitext(export_filenames[0])[0] + ".csv")
