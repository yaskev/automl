from enum import Enum


class ModelType(Enum):
    BAGGED_TREES = 'bagged_trees',
    BOOSTED_TREES = 'boosted_trees',
    LOG_REGRESSION = 'log_regression',
    RANDOM_FOREST = 'random_forest',
    SVM = 'svm'
