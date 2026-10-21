
'''
Stage 2 - MLP on MNIST digit classification
'''

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from local_code.stage_2_code.dataset_loader import Dataset_Loader
from local_code.stage_2_code.method_mlp import Method_MLP
from local_code.stage_2_code.result_saver import Result_Saver
from local_code.stage_2_code.setting_mlp import Setting_MLP
from local_code.stage_2_code.evaluation_metrics import Evaluate_Metrics

# ---- parameter section -----------------------------------
np.random.seed(2)
torch.manual_seed(2)

# ---- object initialization -------------------------------
data_obj = Dataset_Loader('MNIST', '')
data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
data_obj.dataset_train_file_name = 'train.csv'
data_obj.dataset_test_file_name = 'test.csv'

method_obj = Method_MLP('multi-layer perceptron', '')

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_2_result/'
result_obj.result_destination_file_name = 'MLP_prediction_result.pkl'

setting_obj = Setting_MLP('pre-split', '')
evaluate_obj = Evaluate_Metrics('metrics', '')

# ---- baseline run ----------------------------------------
print('************ Start ************')

setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()

scores = setting_obj.load_run_save_evaluate()

print('************ Overall Performance ************')
for key, value in scores.items():
    print(f'  {key:<20}: {value:.4f}')

print('************ Finish ************')

# ---- convergence curves ----------------------------------
os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)

epochs = range(1, len(method_obj.train_loss_history) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, method_obj.train_loss_history, 'b-o', markersize=4)
ax1.set_title('Training Loss Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, [accuracy * 100 for accuracy in method_obj.train_acc_history], 'g-o', markersize=4)
ax2.set_title('Training Accuracy Curve')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim([0, 101])
ax2.grid(True, alpha=0.3)

plt.suptitle('Stage 2 MLP - MNIST Digit Classification')
plt.tight_layout()
plt.savefig(os.path.join(result_obj.result_destination_folder_path, 'learning_curve.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- load data for ablation experiments -------------------
data = data_obj.data

X_train = data['train']['X']
y_train = data['train']['y']
X_test = data['test']['X']
y_test = data['test']['y']

# ---- ablation study ---------------------------------------
print('\n************ Ablation Study ************')

results = []

def ablation_run(name, model):
    torch.manual_seed(2)
    predictions = model.run(X_train, y_train, X_test)
    evaluate_obj.data = {'true_y': y_test, 'pred_y': predictions}
    metric_scores = evaluate_obj.evaluate()

    row = {'Configuration': name, 'Accuracy': metric_scores['Accuracy'], 'Weighted Precision': metric_scores['Weighted Precision'],
           'Weighted Recall': metric_scores['Weighted Recall'],
           'Weighted F1': metric_scores['Weighted F1']}
    
    results.append(row)

    print(f"  {name:<30} "
          f"Accuracy={row['Accuracy']:.4f}  "
          f"F1={row['Weighted F1']:.4f}")


# Baseline
ablation_run('Baseline (512-256-128)', Method_MLP('', ''))

# Shallow architecture
model = Method_MLP('', '')
model.model = nn.Sequential(nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(),
                            nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128),
                            nn.ReLU(), nn.Linear(128, 10))
ablation_run('Shallow (256-128)', model)

# No dropout
model = Method_MLP('', '')
model.model = nn.Sequential(nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(),
                            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Linear(128, 10))
ablation_run('No dropout', model)

# SGD optimizer with momentum
model = Method_MLP('', '')
model.optimizer_class = torch.optim.SGD
model.optimizer_kwargs = {'momentum': 0.9}
ablation_run('SGD + momentum=0.9', model)

# MSE loss function (MSE loss requires one-hot targets instead of class indices)
class MSEMethod(Method_MLP):

    def train_model(self, X, y):
        self.train()
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        loader = DataLoader(TensorDataset(torch.FloatTensor(X),torch.LongTensor(y)), batch_size=self.batch_size, shuffle=True)
        self.train_loss_history = []
        self.train_acc_history = []
        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                targets = nn.functional.one_hot(y_batch, num_classes=10).float()
                loss = self.loss_function(torch.softmax(outputs, dim=1), targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
                correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total += X_batch.size(0)
            self.train_loss_history.append(epoch_loss / total)
            self.train_acc_history.append(correct / total)

model = MSEMethod('', '')
model.loss_function = nn.MSELoss()
ablation_run('MSE loss', model)

# Higher learning rate
model = Method_MLP('', '')
model.learning_rate = 0.01
ablation_run('Learning rate=0.01', model)

# Lower learning rate
model = Method_MLP('', '')
model.learning_rate = 0.0001
ablation_run('Learning rate=0.0001', model)

# Smaller batch size
model = Method_MLP('', '')
model.batch_size = 64
ablation_run('Batch size=64', model)

# Larger batch size
model = Method_MLP('', '')
model.batch_size = 512
ablation_run('Batch size=512', model)

# ---- ablation results summary -----------------------------
print('\n************ Ablation Results Summary ************')

print(
    f"{'Configuration':<30} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10}"
)
print('-' * 74)

for i in results:
    print(f"{i['Configuration']:<30} "
          f"{i['Accuracy']:>10.4f} "
          f"{i['Weighted Precision']:>10.4f} "
          f"{i['Weighted Recall']:>10.4f} "
          f"{i['Weighted F1']:>10.4f}")

print('************ Finished Ablation Study ************')
