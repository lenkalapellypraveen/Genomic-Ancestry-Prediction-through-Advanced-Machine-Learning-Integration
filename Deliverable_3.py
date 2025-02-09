"""For the optimal 𝜆, retrained model on the entire dataset of 𝑁 = 183 observations to obtain an estimate of the (𝑝 + 1) × 𝐾 model parameter matrix as 𝐁̂ and make predictions of the probability for each of the 𝐾 = 5 classes for the 111 test individuals located in TestData_N111_p10.csv."""

import pandas as pd
import numpy as np

class Test:
    def __init__(self, test_data, model, input_standardized, feature_columns, target_one_hot, class_labels, best_lambda):
        self.model = model
        self.input_standardized = input_standardized
        self.feature_columns = feature_columns
        self.target_one_hot = target_one_hot
        self.class_labels = class_labels
        self.best_lambda = best_lambda
        self.test_data = test_data

    def testing(self):
        test_features = self.test_data[self.feature_columns].values
        # Standardizing the test dataset using the training data's mean and standard deviation
        X_test_standardized = (test_features - np.mean(self.input_standardized[:, 1:], axis=0)) / np.std(self.input_standardized[:, 1:], axis=0)
        X_test_augmented = np.column_stack([np.ones(X_test_standardized.shape[0]), X_test_standardized])

        # re-training the model with the best lambda
        self.model.train(self.input_standardized, self.target_one_hot, self.best_lambda)

        # Predicting probabilities for the test dataset
        probabilities_test = self.model.predict(X_test_augmented)

        # Finding predicted labels for each test sample
        self.predicted_labels = np.argmax(probabilities_test, axis=1)

        # Printing the probabilities and predictions for each test sample
        for i, p in enumerate(probabilities_test):
            print(f"Test Sample {i + 1}:")
            print(f"Class Probabilities: {p}")
            print(f"Predicted Ancestry Label: {self.class_labels[self.predicted_labels[i]]}\n")

    def prediction(self):
        # Converting numerical labels to class names
        class_labels = ['African', 'EastAsian', 'European', 'NativeAmerican', 'Oceanian']
        predicted_class_labels = [class_labels[label] for label in self.predicted_labels]

        # Adding predictions to the test dataset
        self.test_data['predicted_class_labels'] = predicted_class_labels

        # Displaying the updated test dataset with predictions
        self.test_data[:20]


