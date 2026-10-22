
"""
Dataset loader for Stage 2 classification.
Loads the training and testing CSV files, separates the features and labels, normalizes the pixel values.
"""

import os
import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        # Load the training and testing datasets
        print("Loading training data...")
        train_data = np.loadtxt(os.path.join(self.dataset_source_folder_path, self.dataset_train_file_name), delimiter=",")

        print("Loading testing data...")
        test_data = np.loadtxt(os.path.join(self.dataset_source_folder_path, self.dataset_test_file_name), delimiter=",")

        # Separate labels from features (pixels) and normalize pixels to [0, 1].
        X_train = train_data[:, 1:].astype(np.float32) / 255.0
        y_train = train_data[:, 0].astype(np.int64)

        X_test = test_data[:, 1:].astype(np.float32) / 255.0
        y_test = test_data[:, 0].astype(np.int64)

        # Keep the train/test split.
        self.data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

        print("Training data shape:", X_train.shape)
        print("Testing data shape:", X_test.shape)

        return self.data

