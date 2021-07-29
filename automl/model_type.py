from enum import Enum
from .models import *


class ModelType(Enum):
    BAGGED_TREES = BaggedTrees
    BOOSTED_TREES = BoostedTrees
    LOG_REGRESSION = LogRegression
    RANDOM_FOREST = RandomForest
    SVM = SVM
