"""Illustrated the effect of the tuning parameter on the inferred ridge regression coefficients by generating five plots (one for each of the 𝐾 = 5 ancestry classes) of 10 lines (one for each of the 𝑝 = 10 features), with the 𝑦-axis as 𝛽̂𝑗𝑘, 𝑗 = 1,2, ... ,10 for the graph of class 𝑘, and 𝑥-axis the corresponding log-scaled tuning parameter value log10(𝜆) that 
generated the particular 𝛽̂𝑗𝑘. Label both axes in all five plots as well as provide a legend for the lines."""

import matplotlib.pyplot as plt
import numpy as np
import os

# generating plots
class PlotGenerator:
    def __init__(self, tuning_parameters, coefficients, feature_labels, class_labels):
        self.tuning_parameters = tuning_parameters
        self.coefficients = coefficients[:, 1:, :]
        self.feature_labels = feature_labels
        self.class_labels = class_labels

    def generate(self, output_dir):
        num_classes = self.coefficients.shape[2]
        num_features = self.coefficients.shape[1]

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(num_classes):
            plt.figure(figsize=(10, 5))
            for j in range(num_features):
                plt.plot(np.log10(self.tuning_parameters),self.coefficients[:, j, i],label=self.feature_labels[j])
            plt.xlabel('Log10(Lambda)')
            plt.ylabel('Ridge Coefficients(Beta)')
            plt.legend()
            plt.title(f'Tuning parameter(Lambda) vs Ridge coefficients: {self.class_labels[i]}')
            plt.grid(True)

            # Construct file path and save the plot
            filename = f"{self.class_labels[i]}_ridge_coefficients.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)

            # Close the plot to free up memory
            plt.close()

