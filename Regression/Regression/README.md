# Regression Model Comparison Project

Welcome to the Regression Model Comparison Project!

This repository provides tools and templates for performing and comparing different regression analyses on a single dataset. The goal is to determine which regression model performs best based on various evaluation metrics.

## Overview

Regression analysis is a key tool in predictive modeling, and choosing the right model is critical for accurate predictions. This project implements and compares the following regression models on a single dataset:

- **Multiple linear Regression**
- **Polynomial Regression**
- **Support Vector Regression (SVR)**
- **Decision Tree Regression**
- **Random Forest Regression**

By applying each of these models to the same dataset, we can evaluate their performance and select the most appropriate model for the data.

## Features

- **Unified Data Pipeline**: A single data pipeline to load, preprocess, and split the dataset for all models.
- **Multiple Regression Implementations**: Templates and code for Polynomial Regression, Support Vector Regression, Decision Tree Regression, and Random Forest Regression.
- **Performance Evaluation**: Tools to calculate and compare metrics such as Mean Squared Error (MSE), R-squared (R²), and Mean Absolute Error (MAE).
- **Visualization Tools**: Visualization of the regression fits and comparison of predicted vs actual values across models.

## Getting Started

### Prerequisites

Ensure you have the following packages installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install the necessary packages using:

```bash
pip install numpy pandas scikit-learn matplotlib 
```

### Dataset

The project uses a single dataset for all regression analyses. Ensure the dataset is in the correct format (CSV or similar) and properly preprocessed for the models.

