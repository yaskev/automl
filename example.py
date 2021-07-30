import pandas as pd

from automl.metric_type import MetricType
from automl.model_selector import ModelSelector
from automl.model_type import ModelType


def get_model_and_metric():
    dataset = pd.read_csv('data/example_dataset.csv')
    labels = dataset.loc[:, 'Survived']
    matrix = dataset.loc[:, dataset.columns != 'Survived']
    selector = ModelSelector(matrix, labels, MetricType.PRECISION)
    selector.turn_off_models(ModelType.LOG_REGRESSION)
    return selector.get_baseline_model(fast_mode=False)


if __name__ == '__main__':
    model, metric = get_model_and_metric()
    print(f'The best model among specified was: {model}')
    print(f'Accuracy: {round(metric, 4)}')
    print(model.get_model_parameters())
