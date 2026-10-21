'''
Saves results to a file.
'''

import os
import pickle

from local_code.base_class.result import result


class Result_Saver(result):

    def __init__(self, rName=None, rDescription=None):
        super().__init__(rName, rDescription)

    def save(self):
        print('Saving results...')

        # Create the result folder if it does not exist
        os.makedirs(self.result_destination_folder_path, exist_ok=True)

        # Combine the folder path and file name
        path = os.path.join(self.result_destination_folder_path, self.result_destination_file_name)

        # Save the results
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

        print('Results saved to:', path)

    def load(self):
        # Load results from the file
        path = os.path.join(self.result_destination_folder_path, self.result_destination_file_name)

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

        return self.data
```
