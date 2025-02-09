import pandas as pd
import numpy as np

# preprocessing the datasets for ridge logistic regression
class DataPreprocessor:
    def __init__(self, filepath, feature_columns, target_column):

        # Loading the dataset.
        self.data = pd.read_csv(filepath)
        self.feature_columns = feature_columns
        self.target_column = target_column

    def standardize_features(self):
        features = self.data[self.feature_columns].values
        standardized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        intercept = np.ones(standardized.shape[0])
        return np.column_stack((intercept, standardized))

    def one_hot_encode_target(self):
        return pd.get_dummies(self.data[self.target_column]).values