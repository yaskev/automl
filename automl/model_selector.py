import pandas as pd

from automl.model_type import ModelType


class ModelSelector:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.models = set(ModelType)

    def get_baseline_model(self):
        pass

    def turn_on_models(self, *args: ModelType):
        pass

    def turn_off_models(self, *args: ModelType):
        pass
