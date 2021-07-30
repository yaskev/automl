import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from automl.models.model import Model


class RandomForest(Model):
    """Wrapper over random forest classifier"""
    def __init__(self, optimize_hyperparams=True):
        super().__init__(RandomForestClassifier(), "Random forest", optimize_hyperparams)

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        grid = {
            'n_estimators': np.arange(10, 151, 10),
            'max_depth': np.arange(3, 16, 3),
            'criterion': ['gini', 'entropy']
        }
        super().fit_with_grid_search(design_matrix, labels, grid)

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return str(self)

