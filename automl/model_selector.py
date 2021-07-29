import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from automl.metric_type import MetricType
from automl.model_type import ModelType


class ModelSelector:
    """This class suggests """
    def __init__(self, design_matrix: pd.DataFrame, labels: np.ndarray, metric: MetricType):
        self.X = design_matrix
        self.y = labels
        self.models = set(ModelType)
        self.encoder = preprocessing.OrdinalEncoder()
        self.scaler = preprocessing.StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.metric = metric

    def get_baseline_model(self):
        self.do_preprocessing()
        best_model = None
        best_metric = 0
        for model in self.models:
            curr_model = model.value(False)
            curr_model.fit(self.X_train, self.y_train)
            curr_metric = curr_model.score(self.X_test, self.y_test, self.metric)
            if curr_metric > best_metric:
                best_model = curr_model
                best_metric = curr_metric

        return best_model, best_metric

    def do_preprocessing(self):
        non_numeric_features = self.X.select_dtypes(exclude='number')
        transformed_features = self.encoder.fit_transform(non_numeric_features)
        self.X = pd.concat([self.X.select_dtypes(include='number'),
                            pd.DataFrame(transformed_features, columns=non_numeric_features.columns)],
                           axis=1)
        self.X = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def turn_on_models(self, *args: ModelType):
        for arg in args:
            if arg in ModelType and arg not in self.models:
                self.models.add(arg)

    def turn_off_models(self, *args: ModelType):
        for arg in args:
            if arg in ModelType and arg in self.models:
                self.models.remove(arg)
