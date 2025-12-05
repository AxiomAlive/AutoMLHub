import itertools
import logging
import multiprocessing
import os
import pprint
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, TypeVar
import numpy as np
import pandas as pd
import openml
from imblearn.datasets import fetch_datasets
from loguru import logger

from data.domain import Dataset
from utils.helpers import make_dataset


class DatasetRepository(ABC):
    def __init__(self, *args, **kwargs):
        self._datasets: List[Dataset] = []

    @abstractmethod
    def load_datasets(self, ids: Optional[List[int]] = None, X_and_y = True) -> List[Dataset]:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, id: int, X_and_y = True) -> Dataset:
        raise NotImplementedError()

    @property
    def datasets(self):
        return self._datasets

# REFACTOR THIS SHIT.
class ZenodoRepository(DatasetRepository):
    def __init__(self):
        super().__init__()
        self._raw_datasets = fetch_datasets(data_home='datasets/imbalanced', verbose=True)

    @logger.catch
    def load_dataset(self, id: int, X_and_y = True) -> Dataset:
        for i, (dataset_name, dataset_data) in enumerate(self._raw_datasets.items(), 1):
            if i == id:
                X = dataset_data.get("data")
                y = dataset_data.get("target")[:, np.newaxis]

                if not X_and_y:
                    X = np.concatenate((X, y), axis=1)
                    y = None
                return make_dataset(
                    name=dataset_name,
                    X=X,
                    y=y
                )
            elif i > id:
                raise ValueError(f"TabularDataset(id={id}) is not available.")

    def load_datasets(self, X_and_y = True, ids: Optional[List[int]] = None) -> List[Dataset]:
        if ids is None:
            range_start = 1
            range_end = len(self._raw_datasets.keys()) + 1
            ids = range(range_start, range_end)
            logger.debug(f"Running tasks from {range_start} to {range_end}.")
        for i in ids:
            self._datasets.append(self.load_dataset(i, X_and_y))
        return self.datasets


# REFACTOR THIS SHIT.
class OpenMLRepository(DatasetRepository):
    def __init__(self, suite_id=271):
        super().__init__()
        self._suite_id = suite_id
        openml.config.set_root_cache_directory("datasets/openml")

    # TODO: parallelize.
    def load_dataset(self, id: Optional[int] = None) -> Dataset:
        task = openml.tasks.get_task(id)
        dataset = task.get_dataset()
        X, y, categorical_indicator, feature_names = dataset.get_data(
            target=dataset.default_target_attribute)

        return make_dataset(
            name=dataset.name,
            y_label=dataset.default_target_attribute,
            X=X,
            y=y
        )

    @logger.catch
    def load_datasets(self, id_range: List[int] = None) -> List[Dataset]:
        benchmark_suite = openml.study.get_suite(suite_id=self._suite_id)
        for i, id in enumerate(benchmark_suite.tasks):
            if id_range is not None and i not in id_range:
                raise ValueError(f"TabularDataset(id={id}) is not available.")
            self._datasets.append(self.load_dataset(id))
        return self.datasets
