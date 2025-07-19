import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union


class SciModels:
    def __init__(self, data: Union[str, pd.DataFrame]):
        """
        Initialize SciModels with a DataFrame or a .pkl file path.
        Args:
            data: Either a pandas DataFrame or a string path to a .pkl file
        """
        if isinstance(data, str):
            self.data_frame = pd.read_pickle(data)
        elif isinstance(data, pd.DataFrame):
            self.data_frame = data
        else:
            raise ValueError("Input must be a DataFrame \
                             or a pickle file path.")

    def train_test_split(self, split: float):
        """
        Perform train/test split on the loaded DataFrame.
        Args:
            split: Fraction to use for test_size in train_test_split
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = self.data_frame['image']
        y = self.data_frame['is_masc_human']
        return train_test_split(X, y, test_size=split, random_state=42)

    # Optionally, add a method to get the splits
    def get_splits(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
