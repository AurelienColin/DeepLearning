import typing

import numpy as np
import pandas as pd
from rignak.custom_display import Display

from src.callbacks.callback import Callback


class HistoryCallback(Callback):

    def __init__(self, *args, batch_size: int, training_steps: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.thumbnail_size: typing.Tuple[int, int] = (12, 6)
        self.ncols: int = 2
        self.x: typing.List[float] = []
        self.logs: typing.Dict[str, typing.Tuple[typing.List[float], typing.List[float]]] = {}

        self.batch_size = batch_size
        self.training_steps = training_steps

    def on_epoch_end(self, epoch: int, logs: typing.Optional[typing.Dict[str, float]] = None) -> None:
        if logs is None:
            logs = {}

        self.x.append((epoch + 1) * self.batch_size * self.training_steps / 1000)

        for i, (key, value) in enumerate(logs.items()):
            base_key = key.replace('val_', '')
            if base_key not in self.logs:
                self.logs[base_key] = [], []
            if base_key == key:
                self.logs[base_key][0].append(value)
            else:
                self.logs[base_key][1].append(value)
            if base_key not in logs:
                self.logs[base_key][0].append(np.nan)
            if 'val_' + base_key not in logs:
                self.logs[base_key][1].append(np.nan)

        try:
            self.model.save(self.model_wrapper.output_folder + "/model.h5", include_optimizer=False)
        except TypeError:
            self.model.save_weights(self.model_wrapper.output_folder + "/model.h5", include_optimizer=False)

        nrows = int(np.ceil(len(self.logs) / self.ncols))
        display = Display(figsize=self.thumbnail_size, nrows=nrows, ncols=self.ncols)

        for i, (key, values) in enumerate(self.logs.items()):
            x = np.array(self.x)
            y = np.array(values)
            display[i].plot(x, y[0], xlabel='kimgs', yscale="log", title=key, labels=key)
            display[i].plot(x, y[1], xlabel='kimgs', yscale="log", title=key, labels='val_' + key)

        display.show(export_filename=self.model_wrapper.output_folder + "/history.png")
        pd.DataFrame(
            {key: value[1] for key, value in self.logs.items()}
        ).to_csv(self.model_wrapper.output_folder + "/history.csv")
