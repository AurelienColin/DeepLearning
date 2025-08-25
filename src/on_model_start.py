import glob
import os
import sys
import typing
import zipfile

import tensorflow as tf


def write_summary(model: tf.keras.models.Model, filename: str) -> None:
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    with open(filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old


def backup(
        output_filename: str, root:
        str = '.',
        patterns: typing.Sequence[str] = ('*.py',),
) -> str:
    filenames = [
        filename
        for pattern in patterns
        for filename in glob.glob(f"{root}/**/{pattern}", recursive=True)
    ]

    with zipfile.ZipFile(output_filename, 'w') as file:
        for filename in filenames:
            file.write(filename, os.path.relpath(filename, root))
    return output_filename
