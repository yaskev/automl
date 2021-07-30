from enum import Enum
from sklearn.metrics import *


class MetricType(Enum):
    ACCURACY = accuracy_score
    BALANCED_ACCURACY = balanced_accuracy_score
    PRECISION = precision_score
    RECALL = recall_score
    ROC_AUC = roc_auc_score
