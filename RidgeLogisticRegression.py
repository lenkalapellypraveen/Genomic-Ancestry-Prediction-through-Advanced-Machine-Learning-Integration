import numpy as np

# implementing Ridge Logistic Regression
class RidgeLogisticRegression:
    def __init__(self, alpha, num_iterations):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.coefficients = None

    def train(self, X, Y, lambda_value):
        num_samples, num_features = X.shape
        num_labels, num_classes = Y.shape

        # Initializing the coefficients
        B = np.zeros((num_features, num_classes))

        # Z matrix for intercepts
        Z = np.copy(B)
        Z[1:, :] = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            Unnormalized_probabilities = np.exp(np.dot(X, B))
            Normalized_probabilities = Unnormalized_probabilities / np.sum(Unnormalized_probabilities, axis=1, keepdims=True)

            # Computing the gradients and updating the coefficients
            gradient = np.dot(X.T, Y - Normalized_probabilities) - 2 * lambda_value * (B - Z)
            B += self.alpha * gradient

        self.coefficients = B
        return B

    def predict(self, X):
        weighted_sum = np.dot(X, self.coefficients)
        exp_values = np.exp(weighted_sum)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities