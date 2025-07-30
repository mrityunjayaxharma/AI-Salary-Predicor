# Salary Predictor & Data Analyzer

A comprehensive Python application that predicts salary levels based on demographic and employment factors using machine learning. Built with Tkinter for the GUI and scikit-learn for the predictive model.

## Features

- **Interactive GUI**: User-friendly interface with multiple tabs for different functionalities
- **Data Visualization**: Built-in visualizations including:
  - Income distribution
  - Age distribution by income
  - Education level impact on income
  - Work hours analysis
  - Correlation matrix
- **Machine Learning**: Implements a Random Forest Classifier for salary prediction
- **Data Exploration**: Tools to explore and understand the dataset
- **Model Evaluation**: Built-in model performance metrics and confusion matrix

## Requirements

- Python 3.6+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, tkinter

## Usage

1. Run [app.py](cci:7://file:///c:/Users/Mrityunjaya%20Sharma/Desktop/ibm%20project/app.py:0:0-0:0) to launch the application
2. Load the dataset (supports both default and custom CSV files)
3. Explore the data visualizations
4. Train the model
5. Make predictions using the prediction tab

## Dataset

The application is designed to work with the Adult Income Dataset (commonly known as "adult.csv"). The dataset should contain demographic and employment-related features used for predicting income levels.

## Note

The application expects the dataset file to be named 'adult.csv' in the same directory, but it will also attempt to load 'adult 3.csv' as a fallback.
