import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np


class DataLoader:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.ord_enc = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_file)
        self.test_data = pd.read_csv(self.test_file)

    def preprocess_data(self):
        # Perform any necessary preprocessing steps on the data
        self.train_data[["Product ID", "Type"]] = self._encode_categorical(
            self.train_data[["Product ID", "Type"]]
        )
        self.test_data[["Product ID", "Type"]] = self._encode_categorical(
            self.test_data[["Product ID", "Type"]]
        )
        self.train_data, self.eval_data = self._split_train_eval(self.train_data)

    def _encode_categorical(self, X):
        if self.ord_enc is None:
            self.ord_enc = OrdinalEncoder(
                unknown_value=100000, handle_unknown="use_encoded_value"
            )
            self.ord_enc.fit(X)
        X_trans = self.ord_enc.transform(X)
        return X_trans

    def _split_train_eval(self, data):
        train_df, eval_df = train_test_split(data, test_size=0.2, random_state=42)
        return train_df, eval_df
