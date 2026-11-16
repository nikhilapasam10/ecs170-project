'''
Evaluation metrics for Stage 4.
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from local_code.base_class.evaluate import evaluate

class Evaluate_Metrics(evaluate):
    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        return {
            'Accuracy': round(accuracy_score(true_y, pred_y), 4),
            'Precision': round(precision_score(true_y, pred_y, zero_division=0), 4),
            'Recall': round(recall_score(true_y, pred_y, zero_division=0), 4),
            'F1': round(f1_score(true_y, pred_y, zero_division=0), 4),
        }
