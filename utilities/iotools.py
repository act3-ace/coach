# Copyright (c) 2023 Mobius Logic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Contains various utility functions/classes used in the project"""
import csv
import json
import os
from collections.abc import Callable  # for type hinting
from typing import Any, Union, cast, NewType  # for type hinting

import numpy as np

EnvId = NewType("EnvId", str)
PathString = str


################################################################################
## Auxiliary Functions
################################################################################
### Turn dictionary with keyed tuple into csv
def save_keyed_tuple(
    dct: dict[tuple[EnvId, EnvId], float],
    filename: PathString,
    do_sort: bool = True,
) -> None:
    """
    Writes a csv file of a dict that is indexed by a tuple (a matrix)

    Parameters
    ----------
    dct : dict
        Dict of values with keys given as tuples
    filename : str
        Name of file to save to
    do_sort : boolean, optional
        whether to sort the output keys
        This is assuming they are of the form "Env_#"

    Side-Effects
    ------------
    None

    Returns
    -------
    None

    Notes
    -----
    As an example, for a dict of:
    dct = {(a,a): val_aa, (a,b): val_ab,
           (b,a): val_ba, (b,b): val_bb,
           (c,a): val_ca, (c,b): val_cb,
           (d,a): val_da, (d,b): val_db}
    The output is:
        ,a,b
        a,val_aa,val_ab
        b,val_ba,val_bb
        c,val_ca,val_cb
        d,val_da,val_db
    Note that the ',' at the start of the first line is intentional
    to provide an empty cell in the csv format so the data forms a
    rectangular matrix.
    """
    # split (l, r) keys into lists of l and r
    l_keys = list({first for first, second in dct})
    r_keys = list({second for first, second in dct})

    # sort by the numeric part of Env_X
    if do_sort:
        l_keys.sort(key=lambda x: int(x.split("_")[1]))
        r_keys.sort(key=lambda x: int(x.split("_")[1]))

    with open(filename, mode="w", newline="", encoding="utf8") as file:
        csvfile = csv.writer(
            file, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        header = [""] + r_keys  # [""] to add blank entry to csv row
        csvfile.writerow(header)

        for first in l_keys:
            row = [first] + [dct.get((first, second), "") for second in r_keys]
            csvfile.writerow(row)


def load_keyed_tuple(
    filename: PathString,
    format_function: Callable[[str], Any] = lambda x: x,
) -> dict[tuple[EnvId, EnvId], Any]:
    """
    Read a tuple-indexed dict (a matrix) from a csv file.

    This is the reverse of save_keyed_tuple
    The format_function is used to convert the values from string to
    whatever format is desirable. The default leaves it as a string.
    A typical choice would be float.

    Parameters
    ----------
    filename : str
        Name of file to save to
    format_function: callable, optional
        This is applied to values (but not keys) read from the file

    Returns
    -------
    dict
        The dict that was read

    Notes
    -----
    For a file with the following:
        ,a,b
        a,val_aa,val_ab
        b,val_ba,val_bb
        c,val_ca,val_cb
        d,val_da,val_db
    The output is:
    dct = {(a,a): val_aa, (a,b): val_ab,
           (b,a): val_ba, (b,b): val_bb,
           (c,a): val_ca, (c,b): val_cb,
           (d,a): val_da, (d,b): val_db}
    """
    dct: dict[tuple[EnvId, EnvId], Any] = {}
    with open(filename, mode="r", newline="", encoding="utf8") as file:
        csvfile = csv.reader(file, delimiter=",")
        # first line has an empty spot to account for alignment
        # of columns. So, we ignore that
        r_keys = next(csvfile)[1:]
        for l_key, *values in csvfile:
            for r_key, value in zip(r_keys, values):
                # appease type checker
                l_key = cast(EnvId, l_key)
                r_key = cast(EnvId, r_key)
                dct[(l_key, r_key)] = format_function(value)
    return dct


################################################################################
## Numpy Encoders for JSON
################################################################################
## Recursively encodes objects with a reprJSON function
# https://stackoverflow.com/questions/5160077/encoding-nested-python-object-in-json
#
# Encdoing numpy objects:
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
#
# Decoding objects
# https://stackoverflow.com/questions/48991911/how-to-write-a-custom-json-decoder-for-a-complex-object

# Usage
# with open(filename, 'w') as jsonfile:
#     json.dump(edge, jsonfile, cls=NumpyEncoder)
# with open(filename, 'r') as jsonfile:
#     edge1 = json.load(jsonfile, cls=NumpyDecoder)


class NumpyEncoder(json.JSONEncoder):
    """Encode numpy data for JSON writer"""

    def default(self, o):  # type: ignore  # this is called by the JSON library
        if isinstance(o, np.integer):
            return {"np.integer": int(o)}
        if isinstance(o, np.floating):
            return {"np.floating": float(o)}
        if isinstance(o, np.ndarray):
            return {"np.array": o.tolist()}
        return json.JSONEncoder.default(self, o)


class NumpyDecoder(json.JSONDecoder):
    """Decode numpy data from JSON"""

    def __init__(self, *args, **kwargs):  # type: ignore  # this is called by the JSON library
        json.JSONDecoder.__init__(self, object_hook=self.numpy_hook, *args, **kwargs)

    def numpy_hook(self, dct):  # type: ignore  # this is called by the JSON library
        """Convert dict with numpy data to numpy object"""
        if "np.integer" in dct:
            return np.int_(dct["np.integer"])
        if "np.floating" in dct:
            return np.float_(dct["np.floating"])
        if "np.array" in dct:
            return np.array(dct["np.array"])
        return dct


################################################################################
## TensorFlow Logging
################################################################################
class TBWriter:
    def __init__(self, args: dict[str, Any]) -> None:
        """ """
        import time

        from tensorflow import summary  # type: ignore[import]

        self.summary = summary
        log_dir = args["log_dir"]

        now = time.localtime()
        subdir = time.strftime("%d-%b-%Y_%H.%M.%S", now)

        self.summary_dir = os.path.join(log_dir, subdir)
        self.summary_writer: dict[str, summary.SummaryWriter] = {}

    def create_scalar_writer(self, name: str) -> None:
        new_dir = os.path.join(self.summary_dir, name)
        self.summary_writer[name] = self.summary.create_file_writer(new_dir)

    def write_item(self, name: str, label: str, data: Any, step: int) -> None:
        if label not in self.summary_writer.keys():
            self.create_scalar_writer(label)

        with self.summary_writer[label].as_default():
            if isinstance(data, (np.ndarray, np.generic)):
                data = data.item()

            if hasattr(data, "__len__"):
                data = data[0]

            self.summary.scalar(name=name, data=data, step=step)
        self.summary_writer[label].flush()

    def write_items(self, name: str, data: dict[str, list[float]], step: int) -> None:
        for label in data.keys():
            self.write_item(name, label, data[label], step)


class Telemetry:
    def __init__(self) -> None:
        self.loggers: list[TBWriter] = []

    def add_logger(self, logger: TBWriter) -> None:
        self.loggers.append(logger)

    def write_item(self, name: str, label: str, data: Any, step: int) -> None:
        for logr in self.loggers:
            logr.write_item(name, label, data, step)

    def write_items(self, name: str, data: dict[str, list[float]], step: int) -> None:
        for logr in self.loggers:
            logr.write_items(name, data, step)
