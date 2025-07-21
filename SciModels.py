import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import os


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

    def train_random_forest(self,
                            X_train, X_test, y_train, y_test,
                            save_name=None):
        """
        Train a RandomForestClassifier and
        print the success rate.
        If save_name is provided, save the model
        to /mnt/ebs_volume/models/<save_name>.
        Returns the trained model.
        """
        model = RandomForestClassifier(random_state=42)
        model.fit(list(X_train), y_train)
        score = model.score(list(X_test), y_test)
        print(f"RandomForestClassifier success rate: {score:.3f}")  # noqa E231
        if save_name:
            save_path = f"/mnt/ebs_volume/models/{save_name}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
            print(f"\U0001F4BE Model saved to {save_path}")
        return model

    def train_svc(self,
                  X_train, X_test, y_train, y_test,
                  save_name=None):
        """
        Train a Support Vector Classifier (SVC) and
        print the success rate.
        If save_name is provided, save the model
        to /mnt/ebs_volume/models/<save_name>.
        Returns the trained model.
        """
        model = SVC(random_state=42)
        model.fit(list(X_train), y_train)
        score = model.score(list(X_test), y_test)
        print(f"SVC success rate: {score:.3f}")  # noqa E231
        if save_name:
            save_path = f"/mnt/ebs_volume/models/{save_name}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
            print(f"\U0001F4BE Model saved to {save_path}")
        return model

    def train_logistic_regression(self,
                                  X_train, X_test, y_train, y_test,
                                  save_name=None):
        """
        Train a LogisticRegression model and print the success rate.
        If save_name is provided, save the model
        to /mnt/ebs_volume/models/<save_name>.
        Returns the trained model.
        """
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(list(X_train), y_train)
        score = model.score(list(X_test), y_test)
        print(f"LogisticRegression success rate: {score:.3f}")  # noqa E231
        if save_name:
            save_path = f"/mnt/ebs_volume/models/{save_name}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
            print(f"\U0001F4BE Model saved to {save_path}")
        return model

    def train_model(self, split_size: float, model: str, save_name=None):
        """
        Wrapper to perform train/test split and train the specified model.
        Args:
            split_size: Fraction to use for test_size in train_test_split
            model: One of 'forest', 'regression', or 'vector'
            save_name: Optional filename to save the trained model
        Returns:
            The trained model
        """
        X_train, X_test, y_train, y_test = self.train_test_split(split_size)
        if model == 'forest':
            return self.train_random_forest(
                X_train, X_test, y_train, y_test, save_name=save_name)
        elif model == 'regression':
            return self.train_logistic_regression(
                X_train, X_test, y_train, y_test, save_name=save_name)
        elif model == 'vector':
            return self.train_svc(
                X_train, X_test, y_train, y_test, save_name=save_name)
        else:
            raise ValueError("Model must be one of: \
                             'forest', 'regression', or 'vector'.")
