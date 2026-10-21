'''
Setting module for pre-split training and testing datasets (no cross-validation)
'''

from local_code.base_class.setting import setting

class Setting_MLP(setting):

    def load_run_save_evaluate(self):
        # Load the training and testing datasets
        data = self.dataset.load()
        X_train = data['train']['X']
        y_train = data['train']['y']
        X_test = data['test']['X']
        y_test = data['test']['y']
        # Train the MLP and predict test labels
        pred_y = self.method.run(X_train, y_train, X_test)

        # Track predictions and true labels
        self.result.data = {'pred_y': pred_y,'true_y': y_test}

        # Result output path and file name
        if self.result.result_destination_folder_path is None:
            self.result.result_destination_folder_path = 'result/stage_2'
        if self.result.result_destination_file_name is None:
            self.result.result_destination_file_name = 'MLP_results.pkl'

        # Save the results
        self.result.save()
        # Evaluate the predictions
        self.evaluate.data = {'pred_y': pred_y,'true_y': y_test}
        return self.evaluate.evaluate()
