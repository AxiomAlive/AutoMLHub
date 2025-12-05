import uuid
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    id: int
    name: str
    X: Union[pd.DataFrame, np.ndarray]
    y: Optional[Union[pd.Series, np.ndarray]] = None
    y_label: Optional[str] = None
    size: Optional[int] = None

@dataclass(frozen=True)
class Task:
    id: int
    dataset: Dataset
    metric: str
