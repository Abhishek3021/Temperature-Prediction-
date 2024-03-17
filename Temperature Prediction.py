#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime

# Load the dataset from a CSV file
file_path = "C:\\Users\\abhis\\OneDrive\\Desktop\\jan2016.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Assuming the dataset has columns 'YEAR', 'MO', 'DY', 'HR', and 'T2M' for year, month, day, hour, and temperature variables
# Extracting features and target variable
X = data[['YEAR', 'MO', 'DY', 'HR']].values  # Assuming 'YEAR', 'MO', 'DY', and 'HR' are the independent variables
y = data['T2M'].values  # Assuming 'T2M' is the dependent variable

# Get user input for prediction
year = int(input("Enter the year (YYYY): "))
month = int(input("Enter the month (1-12): "))
day = int(input("Enter the day (1-31): "))
hour = int(input("Enter the hour (0-23): "))

# List of regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net": ElasticNet(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
    "Support Vector Regression": SVR(),
    "K-Nearest Neighbors Regression": KNeighborsRegressor()
}

# Predict temperature for each model
for name, model in models.items():
    print(f"Predicting temperature with {name}...")
    # Training the model
    model.fit(X, y)
    # Predicting temperature based on user input
    prediction_input = np.array([[year, month, day, hour]])
    temperature_prediction = model.predict(prediction_input)
    print(f"Predicted temperature with {name}: {temperature_prediction[0]}")
    print()

