from collections import OrderedDict
import collections
import abc
import functools
from selectors import EpollSelector
from typing import Callable, List, Mapping
import datasets
import logging
import numpy as np
import torch
import re
import itertools
import os

logger = logging.getLogger(__name__)


from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from typing import List, Dict
from collections import defaultdict
import warnings


from .processor import AbstractTask

main_dir = "/root/xtlv/data/sora_datasets/glue_datasets_from_dn"

##GLUE
class COLA(AbstractTask):
    name = "cola"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_from_disk(f"{main_dir}/cola")[split]


class SST2(AbstractTask):
    name = "sst2"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}




class MRPC(AbstractTask):
    name = "mrpc"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


class QQP(AbstractTask):
    name = "qqp"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class STSB(AbstractTask):
    name = "stsb"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}



class MNLI(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

class MNLI_M(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

class MNLI_MM(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_mismatched"}


class QNLI(AbstractTask):
    name = "qnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


#Tested
class RTE(AbstractTask):
    name = "rte"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class WNLI(AbstractTask):
    name = "wnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


TASK_MAPPING = OrderedDict(
    [
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('mnli-m', MNLI_M),
        ('mnli-mm', MNLI_MM),
        ('qqp', QQP),
        ('stsb', STSB),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, data_args, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, data_args, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )

if __name__ == "__main__":
    for name in TASK_MAPPING:
        print(name)
        task = AutoTask().get(name, None, None)
        print(task.split_train_to_make_test)
        print(task.split_valid_to_make_test)
        train_set = task.get("train", split_validation_test=True)
        print(train_set[0])
