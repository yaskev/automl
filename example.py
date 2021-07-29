import pandas as pd

from automl.metric_type import MetricType
from automl.model_selector import ModelSelector


def get_model_and_metric():
    dataset = pd.read_csv('data/example_dataset.csv')
    labels = dataset.loc[:, 'Survived']
    matrix = dataset.loc[:, dataset.columns != 'Survived']
    selector = ModelSelector(matrix, labels, MetricType.ACCURACY)
    return selector.get_baseline_model()


if __name__ == '__main__':
    model, metric = get_model_and_metric()
    print(model)
    print(f'{MetricType.ACCURACY.value[1]}: {round(metric, 4)}')
