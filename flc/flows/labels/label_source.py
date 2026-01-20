from enum import Enum
from collections import defaultdict
import os
from typing import Optional

import pickle

from .flow_labels import FlowLabel


class LabelSource(Enum):
    CDX2009 = 1
    ISCXVPN2016 = 2
    CICIDS2012 = 3
    CICIDS2017 = 4


class Label:
    def __init__(self, source: LabelSource, label: FlowLabel):
        self.label = label
        self.source = source

    def __str__(self):
        return str(self.label)

    @property
    def flow_label_idx(self):
        return self.label.value


def load_labels(labels_file: str) -> defaultdict[str, Optional[str]]:
    """
    Restores the labels from the given pickle file, or returns an empty dict if file does not exist.
    :param labels_file: path to pickle file that contains previously saved labels
    :return: dict containing filepaths as keys and their label or None if the flow was analyzed, but no label was found
    """
    if os.path.exists(labels_file):
        with open(labels_file, "rb") as fp:
            return defaultdict(list, pickle.load(fp))
    else:
        return defaultdict(list)


def save_labels(labels: dict[str, list[Label]], labels_file: str) -> None:
    """
    Saves the already labeled flows into a pickle file.
    :param labels: dict where keys are filepaths and values are their label
    :param labels_file: file path to the pickle file
    """
    with open(labels_file, "wb") as fp:
        pickle.dump(dict(labels), fp, pickle.HIGHEST_PROTOCOL)
