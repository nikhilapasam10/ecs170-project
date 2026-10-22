'''
Method class for MLP on 10-class classification
(784 pixel features, 10 digit classes)
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method

class Method_MLP(method, nn.Module):

    # Training parameters
    learning_rate = 0.001
    max_epoch = 30
    batch_size = 256

    def __init__(self, mName=None, mDescription=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Set optimizer and its parameters
        self.optimizer_class = torch.optim.Adam
        self.optimizer_kwargs = {}
        # MLP architecture: 784 -> 512 -> 256 -> 128 -> 10
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

        # Cross-entropy for multiclass classification
        self.loss_function = nn.CrossEntropyLoss()
        # Tracks loss and accuracy after each epoch
        self.train_loss_history = []
        self.train_acc_history = []

    def forward(self, x):
        # Pass the input through the network
        return self.model(x)

    def train_model(self, X, y):
        # Enable training mode for dropout and batch normalization
        self.train()
        # Initialize the optimizer
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        # Convert the data into tensors and create batches
        dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loss_history = []
        self.train_acc_history = []

        # Train the model for 30 epochs
        for epoch in range(self.max_epoch):
            epoch_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in loader:
                # Calculate predictions and loss
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = self.loss_function(outputs, y_batch)
                # Backpropagation and updates parameter
                loss.backward()
                optimizer.step()
                # Track loss and correct predictions
                epoch_loss += loss.item() * X_batch.size(0)
                correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total += X_batch.size(0)

            # Average loss and accuracy for this epoch
            avg_loss = epoch_loss / total
            avg_acc = correct / total
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(avg_acc)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:>3}/{self.max_epoch} '
                      f'Loss: {avg_loss:.4f} '
                      f'Acc: {avg_acc:.4f}')

    def test_model(self, X):
        self.eval()
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            # Select the digit with the highest output score
            predictions = outputs.argmax(dim=1)
        return predictions.cpu().numpy()

    def run(self, X_train, y_train, X_test):
        # Train the model, then predict labels for the test data
        print('--start training...')
        self.train_model(X_train, y_train)

        print('--start testing...')
        return self.test_model(X_test)
