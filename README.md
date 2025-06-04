# Stock Price Prediction using RNN

This project implements a **Recurrent Neural Network (RNN)**, specifically using **Long Short-Term Memory (LSTM) layers**, to predict future stock prices. The model learns patterns from historical stock data to forecast upcoming values, demonstrating a fundamental application of deep learning in time series analysis.

## Overview

The goal is to predict the 'Open' price of a stock using its past performance. The model processes sequences of historical stock prices to understand trends and make informed predictions, illustrating how LSTMs are particularly well-suited for sequential data.

## Features

* **Time Series Prediction:** Forecasts future stock prices based on historical data.
* **LSTM Neural Network:** Utilizes multiple layers of LSTMs, which are effective for capturing temporal dependencies in sequential data like stock prices.
* **Data Preprocessing:** Employs Min-Max Scaling to normalize input data, improving model stability and performance.
* **Sliding Window Data Preparation:** Structures the time series data into sequences (timesteps) and corresponding target values for supervised learning.
* **Regression Task:** Uses Mean Squared Error (MSE) as the loss function, appropriate for predicting continuous values.
* **Visualized Results:** Plots actual vs. predicted stock prices for clear comparison.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn (`MinMaxScaler`)

## How It Works

The project involves the following steps:

1.  **Data Loading and Scaling:**
    * Historical stock data is loaded from `Train_set.csv` and `Test_set.csv`.
    * The 'Open' price column, which is the target for prediction, is extracted.
    * Prices are scaled using `MinMaxScaler` to bring them into a range between 0 and 1. This helps the neural network learn more effectively.

2.  **Creating Time Series Sequences:**
    * The scaled training data is transformed into sequences of 60 timesteps (e.g., 60 previous days' prices) to predict the price on the 61st day. These sequences form the `x_train` and `y_train` datasets.

3.  **Model Architecture:**
    * A Sequential Keras model is built with multiple `LSTM` layers.
    * Intermediate LSTM layers use `return_sequences=True` to pass the full sequence output to the next layer.
    * `Dropout` layers are incorporated after each LSTM to prevent overfitting.
    * A final `Dense` layer outputs the single predicted stock price.

4.  **Model Compilation and Training:**
    * The model is compiled with the `adam` optimizer and `mean_squared_error` as the loss function, suitable for regression tasks.
    * The model is then trained on the prepared `x_train` and `y_train` data for a specified number of epochs.

5.  **Prediction and Visualization:**
    * The model makes predictions on a similarly prepared `x_test` dataset (combining the end of training data with test data to ensure correct timesteps).
    * The predicted prices are then inverse-transformed back to their original scale.
    * Finally, `matplotlib` is used to plot the real stock prices against the predicted stock prices, allowing for a visual assessment of the model's forecasting performance.

**Note:** This project requires `Train_set.csv` and `Test_set.csv` files containing historical stock data to be present in the project's root directory. The 'Open' price column is used for training and prediction.
