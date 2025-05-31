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
            display[i_metric].scatter(xs, ys_single_parameter, alpha=0.25, color=colors[j_parameter])
            display[i_metric].scatter(xs, np.mean(ys_single_parameter, axis=1), color=colors[j_parameter],
                                      labels=f"{parameter_value}", title=metric_name)
    display.show(export_filename=export_filename)


def get_xy(
        filenames: typing.Sequence[str]
) -> typing.Tuple[np.ndarray, typing.Dict[str, typing.Dict[typing.Any, typing.List[np.ndarray]]]]:
    xs: np.ndarray = np.arange(20)
    ys: typing.Dict[str, typing.Dict[typing.Any, typing.Sequence[np.ndarray]]] = {}

    for filename in filenames:
        parameter = int(os.path.basename(os.path.dirname(filename)).split('_')[0])
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


def main(pattern: str) -> None:
    filenames = sorted(glob.glob(pattern))
    print(f"{len(filenames)=}")

    xs, ys = get_xy(filenames)

    export_filename = os.path.dirname(os.path.dirname(pattern)) + "/comparison.png"
    plot(xs, ys, export_filename)
