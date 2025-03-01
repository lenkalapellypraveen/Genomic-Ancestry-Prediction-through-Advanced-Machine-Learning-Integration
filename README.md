# Genomic Ancestry Prediction System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

## Project Summary
This project implements a custom Ridge Logistic Regression classifier for genomic ancestry prediction from DNA-derived principal components. By applying advanced regularization techniques and gradient-based optimization, the system achieves classification across five major ancestry groups: African, East Asian, European, Native American, and Oceanian. The implementation includes both a custom-built algorithm and a scikit-learn comparative approach, demonstrating the practical application of machine learning in genomic medicine and population genetics.

## Business Context & Applications

### Precision Medicine
Ancestry information derived from genomic data plays a crucial role in precision medicine, where treatment efficacy and drug response can vary significantly based on genetic ancestry. This model provides a foundational tool for:

- **Pharmacogenomics**: Tailoring medication dosages and selections based on ancestry-related genetic variations
- **Disease Risk Assessment**: Refining risk calculations for conditions with known ancestry-specific prevalence (e.g., sickle cell anemia, Tay-Sachs disease)
- **Clinical Trial Stratification**: Ensuring balanced representation across ancestry groups for more generalizable medical research

### Commercial Applications
Several commercial applications leverage this type of ancestry prediction:

- **Direct-to-Consumer Genetic Testing**: Companies like 23andMe and Ancestry.com use similar approaches to provide heritage estimates
- **Forensic Analysis**: Assisting law enforcement in cases where ancestral information may provide investigative leads
- **Biodiversity Research**: Supporting conservation efforts by mapping genetic diversity across populations

## Technical Framework

### Data Architecture

| Feature | Description |
|---------|-------------|
| PC1 | First principal component of genomic data |
| PC2 | Second principal component of genomic data |
| PC3 | Third principal component of genomic data |
| PC4 | Fourth principal component of genomic data |
| PC5 | Fifth principal component of genomic data |
| PC6 | Sixth principal component of genomic data |
| PC7 | Seventh principal component of genomic data |
| PC8 | Eighth principal component of genomic data |
| PC9 | Ninth principal component of genomic data |
| PC10 | Tenth principal component of genomic data |
| Ancestry | Target variable (African, EastAsian, European, NativeAmerican, Oceanian) |

These principal components represent dimensional reductions of thousands of genetic markers (SNPs), capturing the most significant variation patterns across human populations.

### System Architecture

```
├── DataPreprocessor.py            # Data standardization and encoding pipeline
├── RidgeLogisticRegression.py     # Core ML algorithm implementation
├── Deliverable_1.py               # Coefficient visualization engine
├── Deliverable_2.py               # Cross-validation framework
├── Deliverable_3.py               # Prediction and evaluation system
├── Deliverable_4.py               # Comparative analysis module
├── main.py                        # Orchestration and execution controller
├── TrainingData_N183_p10.csv      # Training dataset (183 samples × 10 features)
├── TestData.csv                   # Validation dataset
└── README.md                      # System documentation
```

## Methodology & Implementation

### Algorithm Design
The project implements a mathematically rigorous approach to multiclass classification with regularization:

#### Mathematical Foundation
The model employs multinomial logistic regression with ridge regularization, optimizing the objective function:

$$\min_{\beta} \left[ -\sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(p_{ik}) + \lambda \sum_{j=1}^{p} \sum_{k=1}^{K} \beta_{jk}^2 \right]$$

Where:
- $y_{ik}$ represents the one-hot encoded target
- $p_{ik}$ is the predicted probability
- $\lambda$ is the regularization strength
- $\beta_{jk}$ are the model coefficients

#### Optimization Strategy
The system employs gradient descent with the following update rule:

```python
gradient = np.dot(X.T, Y - Normalized_probabilities) - 2 * lambda_value * (B - Z)
B += self.alpha * gradient
```

This approach ensures stable convergence while effectively balancing the bias-variance tradeoff through regularization.

### Hyperparameter Tuning
The regularization parameter λ is systematically evaluated through 5-fold cross-validation across a logarithmic scale from 10⁻⁴ to 10⁴, identifying the optimal balance between model complexity and generalization.

### Performance Visualization
The system generates comprehensive visualizations of how feature importance varies with regularization strength, providing interpretable insights into genomic ancestry markers.

## Results & Insights

### Feature Significance Patterns
Analysis of coefficient patterns reveals distinctive genomic signatures for different ancestry groups:

- **African Ancestry**: Strongly characterized by PC1, with sharply negative coefficients at low regularization
- **East Asian Ancestry**: Dominated by PC4, maintaining high positive coefficients even with moderate regularization
- **European Ancestry**: Distinguished primarily by PC2, showing consistent negative values
- **Native American Ancestry**: Identified through a combination of PC1 (positive) and PC3 (negative)
- **Oceanian Ancestry**: Characterized by strong positive PC3 values and moderate PC2 influence

### Regularization Impact
The cross-validation curve demonstrates that:

- Optimal performance is achieved with λ values below 1.0
- Performance degrades rapidly with over-regularization (λ > 10)
- The model shows good stability across a range of small λ values, indicating robust generalization

## Deployment Guidelines

### System Requirements
- Python 3.7+
- Core dependencies: NumPy, Pandas, Matplotlib
- Optional: Scikit-learn (for comparative analysis)

### Installation & Execution
```bash
# Clone repository
git clone https://github.com/username/genomic-ancestry-prediction.git
cd genomic-ancestry-prediction

# Install dependencies
pip install -r requirements.txt

# Run the custom implementation
python main.py

# Run the scikit-learn comparison
python Deliverable_4.py
```

### Integration Scenarios
The model can be integrated into:

1. **Clinical Decision Support Systems**: To inform treatment selection
2. **Research Pipelines**: For population genetics studies
3. **Biobank Analysis Tools**: For large-scale genomic data repositories
4. **Personal Genomics Platforms**: For ancestry reporting services

## Future Development Roadmap

### Planned Enhancements
1. **Algorithmic Improvements**
   - Implement adaptive learning rates for faster convergence
   - Add stochastic gradient descent support for larger datasets
   - Incorporate early stopping based on validation performance

2. **Feature Engineering**
   - Explore non-linear transformations of principal components
   - Investigate interaction terms between significant PCs
   - Implement automated feature selection

3. **Evaluation Framework**
   - Add confusion matrix visualization
   - Implement ROC and precision-recall analysis
   - Develop ancestry-specific performance metrics

4. **Deployment Tools**
   - Create containerized version for portable deployment
   - Develop API interface for service integration
   - Implement batch processing capability for high-throughput scenarios

## Conclusion
This Genomic Ancestry Prediction System demonstrates the successful implementation of a regularized multinomial classification algorithm for population genetics applications. The custom gradient-based approach achieves performance comparable to established libraries while providing deeper insights into the mathematical foundations of the model. The implementation balances technical rigor with practical applicability, making it suitable for both research and commercial applications in the growing field of genomic medicine.

---
© 2025 | Developed as part of advanced machine learning research in computational genomics
