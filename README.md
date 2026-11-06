# Heart Disease Prediction Using Machine Learning

## Project Overview

This project investigates the application of supervised machine learning algorithms to predict the presence of heart disease using health indicators. Using the [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data) dataset, we trained and evaluated three classification models: Perceptron, Random Forest, and Multilayer Perceptron (MLP).

The primary objective was to minimize false-negative predictions by maximizing recall. Hyperparameter optimization and statistical evaluation were performed to compare model performance and identify the most suitable classifier.

## Technical Report

A detailed description of the methodology, model implementations, experimental results, and statistical analysis is available in the **[Final Project Report](Final%20Project%20Report.pdf)**.

## Machine Learning Methods

The following machine learning models and techniques were investigated:

- **Perceptron:** Linear classification model trained using iterative weight updates.
- **Random Forest:** Ensemble learning method combining multiple decision trees.
- **Multilayer Perceptron (MLP):** Feedforward neural network trained using backpropagation.
- **Hyperparameter Optimization:** Model parameter tuning to improve classification performance.
- **Stratified Cross-Validation:** Five-fold cross-validation to evaluate model performance while preserving class distributions.
- **Classification Threshold Optimization:** Threshold selection using Youden's J statistic.

## Model Evaluation and Results

Model performance was evaluated with an emphasis on recall to reduce the frequency of false-negative predictions. Accuracy and precision were also considered to assess the tradeoffs between classification metrics.

| Model | Recall | Accuracy |
|---|---|---|
| Perceptron | 0.88 | 0.68 |
| Random Forest | 0.81 | 0.76 |
| MLP | 0.80 | 0.77 |

Following hyperparameter optimization and statistical comparisons, the Perceptron was selected as the preferred model based on the recall-focused evaluation criteria.

## Implementation

The Jupyter notebook containing the machine learning implementation and model evaluation is available below:

**[Machine Learning Implementation](MLPModel.ipynb)**

The project uses Python and machine learning libraries including NumPy, pandas, scikit-learn, and PyTorch.

## Dataset

The dataset was obtained from Kaggle:

[Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)

The dataset contains survey-based health indicators associated with heart disease. Data preprocessing was performed prior to model training and evaluation to address data quality and class imbalance.
