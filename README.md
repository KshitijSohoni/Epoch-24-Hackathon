# Epoch-24-Hackathon

## 
This project was developed for a machine learning hackathon, where it secured the second-place position. The model's performance and innovative approach contributed to its recognition in the competition.


## Overview

This repository contains code for a deep learning model that addresses a classification task. The project involves predicting certain medical conditions based on patient data.

## Preprocessing

1. **Loading Data**: The data is loaded into a Pandas DataFrame from the provided CSV files (`train.csv` and `test.csv`).
2. **Handling Missing Values**: Missing values denoted by '?' are replaced with Pandas' missing value representation (`pd.NA`).
3. **Outlier Handling**: Outliers in numerical features are treated using the Interquartile Range (IQR) method. Values outside 1.5 times the IQR are replaced with random values generated within a specified range.
4. **Imputation**: Missing values in numerical features are imputed with the mean of the respective columns.
5. **Label Encoding**: Categorical variables are encoded using Label Encoding to convert them into numerical format for model training.

## Model Building
1. **Feature Selection**: Certain columns are dropped based on domain knowledge or feature importance analysis.
2. **Data Splitting**: The dataset is split into training and testing sets using a 80:20 ratio.
3. **Feature Scaling**: Standardization is applied to numerical features to bring them to a common scale.
4. **Deep Learning Model Architecture**: A deep neural network model is constructed using TensorFlow's Keras API. The architecture includes multiple Dense layers with ReLU activation, Dropout layers for regularization, and L2 regularization to prevent overfitting.
   
   - **Input Layer**: The input layer has neurons corresponding to the number of features in the dataset.
   - **Hidden Layers**: Multiple dense layers with ReLU activation functions are added to extract complex patterns from the data.
   - **Dropout Layers**: Dropout layers are included to prevent overfitting by randomly dropping a fraction of input units.
   - **Output Layer**: The output layer has neurons equal to the number of target variables with sigmoid activation for binary classification.
   
6. **Model Training**: The model is trained using the training data with early stopping to prevent overfitting.

   
## Evaluation and Prediction
1. **Evaluation Metrics**: The model's performance is evaluated using accuracy score on the test set.
2. **Prediction**: The trained model is then used to predict the target variables on the provided test dataset.


## Requirements
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Scikit-learn
- TensorFlow
