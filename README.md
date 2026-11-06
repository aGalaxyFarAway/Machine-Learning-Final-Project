# Heart Disease Prediction Using Machine Learning

## Project Overview

This project investigates the application of supervised machine learning algorithms to classify heart disease using health indicators. Using the [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data) dataset, three classification models were trained and evaluated: Perceptron, Random Forest, and Multilayer Perceptron (MLP).

The primary objective was to minimize false-negative predictions by maximizing recall. Systematic hyperparameter tuning and statistical evaluation were performed to compare model performance and investigate the tradeoffs between recall, precision, and overall classification accuracy.

## Technical Report

A detailed description of the methodology, model implementations, hyperparameter optimization, experimental results, and statistical analysis is available in the **[Final Project Report](Final%20Project%20Report.pdf)**.

## Dataset and Preprocessing

The project utilized the 2020 version of the Personal Key Indicators of Heart Disease dataset, containing 319,795 observations collected from CDC health surveys.

The preprocessing procedure included:

- Random undersampling of the majority class to obtain a balanced dataset of 54,746 observations.
- Binary encoding of categorical features and one-hot encoding of multiclass features.
- Correlation analysis using Pearson's correlation coefficient to investigate multicollinearity.
- Feature standardization using `StandardScaler`.
- Five-fold stratified cross-validation, with scaling parameters fitted exclusively on training subsets to prevent data leakage.

## Machine Learning Methods

Three supervised classification algorithms were investigated:

- **Perceptron:** Linear binary classifier with hyperparameter tuning of learning rate, regularization, convergence tolerance, and maximum iterations.
- **Random Forest:** Ensemble classifier with hyperparameter tuning of tree depth, feature sampling, and minimum samples per leaf.
- **Multilayer Perceptron (MLP):** Feedforward neural network implemented in PyTorch with two hidden layers, ReLU activations, binary cross-entropy loss with logits, and Adam optimization.

Model performance was evaluated using recall, precision, F1-score, accuracy, and ROC-AUC.

## Model Evaluation and Results

The following results were obtained after hyperparameter optimization and evaluation on the held-out test set.

| Model | Recall | Precision | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Perceptron | 0.88 | 0.63 | 0.68 | 0.785 |
| Random Forest | 0.81 | 0.73 | 0.76 | 0.83 |
| MLP | 0.80 | 0.75 | 0.77 | 0.841 |

The Perceptron achieved the highest recall, consistent with the objective of minimizing false-negative classifications. However, this improvement came at the expense of precision and overall accuracy.

Random Forest and MLP demonstrated more balanced classification performance. Considering both predictive performance and ease of implementation, Random Forest was recommended as the preferred model for this classification task.

## Individual Contributions

My contributions to this group project focused on data preprocessing and the implementation and evaluation of the multilayer perceptron.

**Data Preprocessing**
- Processed and encoded categorical and numerical features for machine learning applications.
- Addressed class imbalance through random undersampling.
- Performed correlation analysis and feature standardization.
- Prepared stratified cross-validation folds while preventing data leakage during feature scaling.

**Neural Network Implementation**
- Implemented a multilayer perceptron using PyTorch.
- Configured the network architecture, loss function, and Adam optimizer.
- Performed hyperparameter tuning across learning rates, hidden-layer sizes, and weight decay values.
- Evaluated model performance using classification metrics, confusion matrices, and ROC curves.
- Applied threshold optimization using the median Youden's J statistic across validation folds.

## Implementation

The Jupyter notebook containing the preprocessing and MLP implementation is available here:

**[Preprocessing and MLP Implementation](MLPModel.ipynb)**

### Technologies Used

- Python
- PyTorch
- scikit-learn
- pandas
- NumPy
- Matplotlib

## Project Authors

Andrew White, David Faik, and Assaf Izhar.
