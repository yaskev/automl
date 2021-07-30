import numpy as np
import pandas as pd
from sklearn.svm import SVC
from automl.models.model import Model


class SVM(Model):
    """Wrapper over support vector machine"""
    def __init__(self, optimize_hyperparams=True):
        super().__init__(SVC(), "Support vector machine", optimize_hyperparams)

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        grid = {
            'C': np.linspace(0.01, 10, 100),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        super().fit_with_grid_search(design_matrix, labels, grid)

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return str(self)

