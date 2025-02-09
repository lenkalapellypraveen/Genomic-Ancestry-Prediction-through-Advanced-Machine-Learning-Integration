'''Implemented this project using statistical or machine learning libraries. Compared the results with the project implemented manually'''

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

class DataLoader:
    def __init__(self, filepath, feature_columns, target_column):
        self.data = pd.read_csv(filepath)
        self.feature_columns = feature_columns
        self.target_column = target_column

    def get_features_and_target(self):
        X = self.data[self.feature_columns].values
        y = self.data[self.target_column].values
        return X, y

# implementing Ridge Logistic Regression
class RidgeLogisticRegression:
    def __init__(self, lambda_values, max_iterations=10000, random_state=42):
        self.lambda_values = lambda_values
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.coefficients = []

    def fit(self, X, y):
        for l in self.lambda_values:
            model = LogisticRegression(penalty='l2', C=1 / l, solver='lbfgs', max_iter=self.max_iterations,
                                       random_state=self.random_state, multi_class='multinomial')
            model.fit(X, y)
            self.coefficients.append(model.coef_.T)
        self.coefficients = np.array(self.coefficients)

        return self.coefficients

# Visualization
class CoefficientPlotter:
    def __init__(self, regularization_params, coefficients, feature_labels, class_labels):
        self.regularization_params = regularization_params
        self.coefficients = coefficients
        self.feature_labels = feature_labels
        self.class_labels = class_labels

    def generate_plots(self):
        num_classes = self.coefficients.shape[2]
        num_features = self.coefficients.shape[1]

        for c in range(num_classes):
            plt.figure(figsize=(10, 5))
            for f in range(num_features):
                plt.plot(np.log10(self.regularization_params), self.coefficients[:, f, c],
                         label=self.feature_labels[f])
            plt.xlabel('Log10(Lambda)')
            plt.ylabel('Ridge Coefficients(Beta)')
            plt.legend()
            plt.title(f'Tuning parameter(Lambda) vs Ridge coefficients: {self.class_labels[c]}')
            plt.grid(True)
            plt.show()

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Tuning parameters
    lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]

    feature_columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
    target_column = 'Ancestry'
    class_labels = ['African', 'EastAsian', 'European', 'NativeAmerican', 'Oceanian']
    feature_labels = feature_columns

    # Loading the training data
    data_loader = DataLoader(filepath=r'TrainingData.csv',
                             feature_columns=feature_columns, target_column=target_column)
    X, y = data_loader.get_features_and_target()

    # Training using ridge logistic regression
    ridge_lr = RidgeLogisticRegression(lambda_values=lambda_values)
    coefficients = ridge_lr.fit(X, y)

################ Deliverable - 1
plotter = CoefficientPlotter(lambda_values, coefficients, feature_labels, class_labels)
plotter.generate_plots()

################ Deliverable - 2 (5-fold cross-validation)
# performing k-fold cross-validation for Ridge Logistic Regression.
scaler = StandardScaler()
class CrossValidator:
    def __init__(self, lambda_values, alpha, n_folds=5, random_state=42):
        self.lambda_values = lambda_values
        self.alpha = alpha
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        self.best_lambda = None
        self.min_avg_error = float('inf')
        self.cv_errors_lb = []
        self.n_folds = n_folds

    def perform_cross_validation(self, X, y):
        for L in self.lambda_values:
            cv_fold_errors = []

            for f, (train_index, val_index) in enumerate(self.kf.split(X)):
                # Splitting the data for the current fold using the indices
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Standardize the data
                X_train_std = scaler.fit_transform(X_train)
                X_val_std = scaler.transform(X_val)

                # Training the logistic regression model
                lr_model = LogisticRegression(C=1 / L, tol=self.alpha, solver='lbfgs', max_iter=10000)
                lr_model.fit(X_train_std, y_train)

                # Predict probabilities for the validation set
                predictions = lr_model.predict_proba(X_val_std)

                # Calculating Mean-Squared-Error for the current fold
                cv_fold_mse = log_loss(y_val, predictions)
                cv_fold_errors.append(cv_fold_mse)

            # calculating average error for the current lambda
            self.avg_error = np.mean(cv_fold_errors)
            self.cv_errors_lb.append(self.avg_error)

            # Updating the best lambda if the current lambda is better
            if self.avg_error < self.min_avg_error:
                self.min_avg_error = self.avg_error
                self.best_lambda = L

        return cv_fold_errors, self.best_lambda, self.min_avg_error

    def plot_cv_errors(self):
        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(np.log10(self.lambda_values), self.cv_errors_lb, marker='d')
        plt.xlabel('Log10(Lambda)')
        plt.ylabel('CV(5) Error')
        plt.title('Tuning parameter(Lambda) vs Cross-Validation Error')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Learning rate
    alpha = 10 ** (-5)

    # Tuning parameters
    lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]

    # Perform cross-validation
    cv = CrossValidator(lambda_values, alpha)
    cv_errors_lb, best_lambda, min_avg_error = cv.perform_cross_validation(X, y)

    # Plot the cross-validation errors
    cv.plot_cv_errors()

print(f'𝜆 value that generated the smallest CV(5) error is: {best_lambda}')
print(f'smallest CV(5) error for Best λ is: {min_avg_error}')

################ Deliverable - 3 (Testing and Prediction)
data_loader = DataLoader(filepath='TestData.csv',
                             feature_columns=feature_columns, target_column=target_column)
Xtest, ytest = data_loader.get_features_and_target()

X_test_augmented_lb = (Xtest - np.mean(Xtest, axis=0)) / np.std(Xtest, axis=0)

# Retraining the training set with the optimal lambda.
lr_model_final = LogisticRegression(C=1/best_lambda,tol=alpha, solver='lbfgs')
lr_model_final.fit(X,y)

# Predicting probabilities for the test dataset
probabilities_test_lb = lr_model_final.predict_proba(X_test_augmented_lb)

# Finding predicted labels for each test sample
predicted_labels_lb = np.argmax(probabilities_test_lb, axis=1)

# Printing the probabilities and predictions for each test sample
for i, p in enumerate(probabilities_test_lb):
    print(f"Test Sample {i + 1}:")
    print(f"Class Probabilities: {p}")
    print(f"Predicted Ancestry Label: {class_labels[predicted_labels_lb[i]]}\n")

# Loading the TestData_N111_p10 dataset
test_data_lb = pd.read_csv('TestData.csv')
test_features_lb = test_data_lb[feature_columns].values

# Converting numerical labels to class names
class_labels_lb = ['African', 'EastAsian', 'European', 'NativeAmerican', 'Oceanian']
predicted_class_labels_lb = [class_labels_lb[label] for label in predicted_labels_lb]

# Adding predictions to the test dataset
test_data_lb['predicted_class_labels_lb'] = predicted_class_labels_lb

# Displaying the updated test dataset with predictions
test_data_lb[:20]
