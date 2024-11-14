import torch
import numpy as np
from sklearn.metrics import f1_score


class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'emotake': self.eval_emotake,
            'other': None
        }

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def eval_emotake(self, y_pred, y_true):
        _, predicted = torch.max(y_pred, 1)
        predicted = torch.tensor(predicted)

        correct = predicted.eq(y_true.data).sum()
        total = len(y_true)
        accuracy = correct / total

        f1 = f1_score(y_true, predicted, average='weighted')

        eval_results = {
            "Accuracy": round(accuracy * 100, 4),
            "F1-Score": round(f1 * 100, 4)
        }
        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
