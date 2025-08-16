import glob
import os
import typing

import numpy as np
import pandas as pd
from rignak.custom_display import Display


def plot(
        xs: np.ndarray,
        ys: typing.Dict[str, typing.Dict[typing.Any, typing.List[np.ndarray]]],
        export_filename: str
) -> None:
    display = Display(ncols=len(ys))

    colors = ('tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:green')
    for i_metric, (metric_name, ys_single_loss) in enumerate(ys.items()):
        for j_parameter, (parameter_value, ys_single_parameter) in enumerate(ys_single_loss.items()):
            ys_single_parameter = np.array(ys_single_parameter).T
            print(f"{parameter_value=}, {ys_single_parameter.shape=}")
            color = colors[j_parameter]
            if ys_single_parameter.shape[0]:
                middle = np.nanmedian(ys_single_parameter, axis=1)
                first_quartile = np.nanpercentile(ys_single_parameter, 25, axis=1)
                third_quartile = np.nanpercentile(ys_single_parameter, 75, axis=1)
                display[i_metric].ax.fill_between(xs, first_quartile, third_quartile, alpha=0.25, color=color)
                display[i_metric].plot(xs, middle, linestyle="dashed", color=color, labels=str(parameter_value), ylabel=metric_name, xlabel="epoch")


    display.show(export_filename=export_filename)


def get_xy(
        filenames: typing.Sequence[str],
        n_epochs: int
) -> typing.Tuple[np.ndarray, typing.Dict[str, typing.Dict[typing.Any, typing.List[np.ndarray]]]]:
    xs: np.ndarray = np.arange(n_epochs)
    ys: typing.Dict[str, typing.Dict[typing.Any, typing.Sequence[np.ndarray]]] = {}

    for filename in filenames:
        parameter = os.path.basename(os.path.dirname(filename)).split('_')[0]
        dataframe = pd.read_csv(filename)
        for loss_name in dataframe.columns[1:]:
            values = dataframe[loss_name].to_numpy()
            if loss_name not in ys:
                ys[loss_name] = {}
            if parameter not in ys[loss_name]:
                ys[loss_name][parameter] = []
            if values.shape[0] == xs.shape[0]:
                ys[loss_name][parameter].append(values)
    return xs, ys


def main(pattern: str, n_epochs: int) -> None:
    filenames = sorted(glob.glob(pattern))
    xs, ys = get_xy(filenames, n_epochs)

    export_filename = os.path.dirname(os.path.dirname(pattern)) + "/comparison.png"
    plot(xs, ys, export_filename)
