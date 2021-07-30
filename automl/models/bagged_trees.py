import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from automl.models.model import Model


class BaggedTrees(Model):
    """Wrapper over bagging classifier"""
    def __init__(self, optimize_hyperparams=True):
        super().__init__(BaggingClassifier(), "Bagged trees", optimize_hyperparams)

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        grid = {
            'n_estimators': np.arange(5, 21, 5)
        }
        super().fit_with_grid_search(design_matrix, labels, grid)

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return str(self)

