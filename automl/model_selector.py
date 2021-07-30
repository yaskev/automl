import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from automl.metric_type import MetricType
from automl.model_type import ModelType


class ModelSelector:
    """This class suggests the fitted model, best among the given set of models"""
    def __init__(self, design_matrix: pd.DataFrame, labels: np.ndarray, metric: MetricType,
                 encoder=preprocessing.OrdinalEncoder(),
                 scaler=preprocessing.StandardScaler()):
        self.X = design_matrix
        self.y = labels
        self.models = list(ModelType)
        self.encoder = encoder
        self.scaler = scaler
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.metric = metric
        self.best_model = None
        self.best_metric = 0

    def get_baseline_model(self, fast_mode: bool = False, verbose: bool = True):
        if self.best_model is None:
            self.__do_preprocessing()
            for model in self.models:
                try:
                    curr_model = model.value(not fast_mode)
                    if verbose:
                        print(f'Fitting {curr_model}...')
                    curr_model.fit(self.X_train, self.y_train)
                    if verbose:
                        print(f'Fitted')
                    curr_metric = curr_model.score(self.X_test, self.y_test, self.metric)
                    if curr_metric > self.best_metric:
                        self.best_model = curr_model
                        self.best_metric = curr_metric
                except Exception as e:
                    if verbose:
                        print(str(e))

        return self.best_model, self.best_metric

    def __do_preprocessing(self):
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
                self.models.append(arg)
                self.best_model = None
                self.best_metric = 0

    def turn_off_models(self, *args: ModelType):
        for arg in args:
            if arg in ModelType and arg in self.models:
                self.models.remove(arg)
                self.best_model = None
                self.best_metric = 0
