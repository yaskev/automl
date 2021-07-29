from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from automl.metric_type import MetricType


class Model(ABC):
    @abstractmethod
    def __init__(self, optimize_hyperparams=True):
        self.optimize_hyperparams = optimize_hyperparams
        self.fitted = False
        self.optimized_model = None

    @abstractmethod
    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        pass

    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def score(self, test_data: pd.DataFrame, test_labels: np.ndarray, metric: MetricType) -> float:
        pass
