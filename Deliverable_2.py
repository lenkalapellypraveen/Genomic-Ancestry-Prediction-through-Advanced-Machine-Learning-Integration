"""Illustrated the effect of the tuning parameter on the cross validation error by generating a plot with the 𝑦-axis as CV(5) error, and the 𝑥-axis the corresponding log-scaled tuning parameter value log10(𝜆) that generated the particular CV(5) error. Labeled both axes in the plot."""

import matplotlib.pyplot as plt
import numpy as np
import os

# performing k-fold cross-validation for Ridge Logistic Regression.
class CrossValidation:
    def __init__(self, model, folds=5, max_iterations=10000):
        self.model = model
        self.folds = folds
        self.max_iterations = max_iterations

    def calculate_mse(self, y_true, y_pred):
        mse = np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
        return mse

    def split_data(self, X, Y, fold_idx):
        fold_size = len(X) // self.folds
        start = fold_idx * fold_size
        end = start + fold_size

        X_val, Y_val = X[start:end], Y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        Y_train = np.concatenate([Y[:start], Y[end:]])

        return X_train, Y_train, X_val, Y_val

    def cross_validation(self, X, Y, lambdas, output_dir):
        cv_errors = []
        best_lambda = None
        min_avg_error = float('inf')

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for L in lambdas:
            cv_fold_errors = []

            for f in range(self.folds):
                # Splitting the data for the current fold
                X_train, Y_train, X_val, Y_val = self.split_data(X, Y, f)

                # Training the model on the training set
                self.model.train(X_train, Y_train, L)

                # Predicting the probabilities for the validation set
                predictions = self.model.predict(X_val)

                # Calculating Mean-Squared-Error for the current fold
                cv_fold_mse = self.calculate_mse(Y_val, predictions)
                cv_fold_errors.append(cv_fold_mse)

            # calculating average error for the current lambda
            avg_error = np.mean(cv_fold_errors)
            cv_errors.append(avg_error)

            # Updating the best lambda if the current lambda is better
            if avg_error < min_avg_error:
                min_avg_error = avg_error
                best_lambda = L

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(np.log10(lambdas), cv_errors, marker='d')
        plt.xlabel('Log10(Lambda)')
        plt.ylabel('CV(5) Error')
        plt.title('Tuning parameter(Lambda) vs Cross-Validation Error')
        plt.grid(True)
        
        # Construct file path and save the plot
        filename = "Tuningparameter(Lambda)_vs_CrossValidationError.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)

        # Close the plot to free up memory
        plt.close()

        return cv_errors, best_lambda, min_avg_error

        
