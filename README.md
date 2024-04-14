#  Diamond Price Prediction
## Overview
This project aims to predict diamond prices based on their physical dimensions (length X, width Y, and height Z). We utilize several machine learning models to estimate the price based on these features.

## Data
The dataset used in this project, diamonds.csv, is sourced from Kaggle and includes features such as carat weight, cut quality, color, clarity, depth, table, and the physical dimensions (X, Y, Z) along with the price.

## Installation
Before running the project, ensure you have Python installed along with the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
You can install them using pip:
`pip install pandas numpy matplotlib seaborn scikit-learn scipy`

## Usage
Load and clean the data as follows:
`import pandas as pd`
`diamonds = pd.read_csv('/content/sample_data/diamonds.csv', index_col=0)`
`diamonds = diamonds[diamonds['x'] * diamonds['y'] * diamonds['z'] != 0] ` # Removing non-physical entries

Visualize data to understand relationships and clean further if necessary. For example:
`import seaborn as sns`
`sns.pairplot(diamonds.select_dtypes(include=['float64', 'int64']))`

## Models
We experimented with the following models:

- Linear Regression
- Decision Tree Regressor
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Each model is evaluated using Root Mean Square Logarithmic Error (RMSLE) to ensure accurate predictions.

## Results
The models provide varying levels of accuracy, with detailed insights into feature importance and model performance. The project explores the impact of different features on the price, emphasizing the importance of carat weight as a significant predictor.

## Conclusion
The project highlights the capability of machine learning techniques in predicting diamond prices based on physical properties and other characteristics. Future improvements could include more sophisticated feature engineering, parameter tuning, and exploring deep learning models.

## Credits
Project developed by [Batel Yerushalmi](http://github.com/BatelCohen7 "Batel Yerushalmi") and [Elchai Agassi](http://github.com/ElhaiAgassi "Elchai Agassi"). Dataset sourced from Kaggle's Diamonds Dataset.

