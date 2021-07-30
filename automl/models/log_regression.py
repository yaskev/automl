import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from automl.models.model import Model


class LogRegression(Model):
    """Wrapper over logistic regression"""
    def __init__(self, optimize_hyperparams=True):
        basic_model = LogisticRegression(multi_class='ovr', solver='saga', tol=1e-3, n_jobs=-1)
        super().__init__(basic_model, "Logistic regression", optimize_hyperparams)

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        grid = {
            'C': np.linspace(0.01, 10, 100)
        }
        super().fit_with_grid_search(design_matrix, labels, grid)

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return str(self)
