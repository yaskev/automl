import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from automl.metric_type import MetricType
from automl.models.model import Model


class BoostedTrees(Model):
    def __init__(self, optimize_hyperparams=True):
        super().__init__(optimize_hyperparams)
        self.basic_model = GradientBoostingClassifier()

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        if self.optimize_hyperparams:
            grid = {
                'max_depth': np.arange(3, 10, 3),
                'n_estimators': np.arange(20, 151, 10)
            }
            self.optimized_model = GridSearchCV(self.basic_model, grid, n_jobs=-1)
        else:
            self.optimized_model = self.basic_model
        self.optimized_model.fit(design_matrix, labels)
        self.fitted = True

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise Exception(f'The model "BoostedTrees" has not been fitted')
        return self.optimized_model.predict(test_data)

    def score(self, test_data: pd.DataFrame, test_labels: np.ndarray, metric: MetricType) -> float:
        if not self.fitted:
            raise Exception(f'The model "BoostedTrees" has not been fitted')
        return metric.value[0](test_labels, self.optimized_model.predict(test_data))

    def __str__(self):
        return 'Boosted trees'

    def __repr__(self):
        return str(self)

