import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

from automl.metric_type import MetricType
from automl.models.model import Model


class BaggedTrees(Model):
    def __init__(self, optimize_hyperparams=True):
        super().__init__(optimize_hyperparams)
        self.basic_model = BaggingClassifier()

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        if self.optimize_hyperparams:
            grid = {
                'n_estimators': np.arange(5, 21, 5)
            }
            self.optimized_model = GridSearchCV(self.basic_model, grid, n_jobs=-1)
        else:
            self.optimized_model = self.basic_model
        self.optimized_model.fit(design_matrix, labels)
        self.fitted = True

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise Exception(f'The model "BaggedTrees" has not been fitted')
        return self.optimized_model.predict(test_data)

    def score(self, test_data: pd.DataFrame, test_labels: np.ndarray, metric: MetricType) -> float:
        if not self.fitted:
            raise Exception(f'The model "BaggedTrees" has not been fitted')
        return metric.value[0](test_labels, self.optimized_model.predict(test_data))

    def __str__(self):
        return 'Bagged trees'

    def __repr__(self):
        return str(self)

