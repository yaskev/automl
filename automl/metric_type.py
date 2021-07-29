from enum import Enum
from sklearn.metrics import *


class MetricType(Enum):
    ACCURACY = (accuracy_score, 'Accuracy')
    BALANCED_ACCURACY = (balanced_accuracy_score, 'Balanced accuracy')
    PRECISION = (precision_score, 'Precision')
    RECALL = (recall_score, 'Recall')
    ROC_AUC = (roc_auc_score, 'ROC-AUC')
