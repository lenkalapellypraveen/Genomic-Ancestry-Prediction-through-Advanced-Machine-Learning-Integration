# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataPreprocessor import DataPreprocessor
from RidgeLogisticRegression import RidgeLogisticRegression
from Deliverable_1 import PlotGenerator
from Deliverable_2 import CrossValidation
from Deliverable_3 import Test

if __name__ == "__main__":
    # Learning rate
    alpha = 10 ** (-5)

    # Tuning parameters
    lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]

    # number of iterations
    num_iterations = 10000

    feature_columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
    target_column = 'Ancestry'
    class_labels = ['African', 'EastAsian', 'European', 'NativeAmerican', 'Oceanian']
    feature_labels = feature_columns

    # loading the dataset, standardizing the features, and one-hot encoding the target variable
    preprocessor = DataPreprocessor(r'TrainingData.csv', feature_columns, target_column)
    input_standardized = preprocessor.standardize_features()
    target_one_hot = preprocessor.one_hot_encode_target()

    # Training the ridge logistic regression model for different lambda values
    model = RidgeLogisticRegression(alpha=alpha, num_iterations=num_iterations)
    ridge_coefficients = np.zeros((len(lambda_values), input_standardized.shape[1], target_one_hot.shape[1]))

    for i, L in enumerate(lambda_values):
        ridge_coefficients[i] = model.train(input_standardized, target_one_hot, L)

    # Deliverable - 1
    plotter = PlotGenerator(lambda_values, ridge_coefficients, feature_labels, class_labels)
    output_dir_1 = r'Deliverable_1_outputs'
    plotter.generate(output_dir_1)

    # Deliverable - 2 (5-fold cross-validation)
    cross_validator = CrossValidation(model, folds=5, max_iterations=10000)
    output_dir_2 = r'Deliverable_2_outputs'
    cv_errors, best_lambda, min_avg_error = cross_validator.cross_validation(input_standardized, target_one_hot, lambda_values, output_dir_2)
    print(f'𝜆 value that generated the smallest CV(5) error is: {best_lambda}')
    print(f'smallest CV(5) error for Best λ is: {min_avg_error}')

    # Deliverable - 3 (Testing and Prediction)
    test_data = pd.read_csv(r'TestData.csv')
    test = Test(test_data, model, input_standardized, feature_columns, target_one_hot, class_labels, best_lambda)
    test.testing()
    test.prediction()
   

    

    









