'''
Accuracy, Weighted Precision, Weighted Recall, Weighted F1
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from local_code.base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):

    def __init__(self, eName=None, eDescription=None):
        super().__init__(eName, eDescription)

    def evaluate(self):
        print('Evaluating performance...')
        #True and predicted labels 
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        #Calculates accuracy, weighted precision, weighted recall, and weighted F1-score
        accuracy  = accuracy_score(true_y, pred_y)
        w_precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        w_recall    = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        w_f1        = f1_score(true_y, pred_y, average='weighted', zero_division=0)
        #Returns the evaluation metrics 
        return {'Accuracy': round(accuracy, 4), 'Weighted Precision': round(w_precision, 4), 'Weighted Recall': round(w_recall, 4),'Weighted F1': round(w_f1,4)}
