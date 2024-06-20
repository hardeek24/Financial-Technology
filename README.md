# Stock-Price-Prediction-Using-LSTM
This repository contains code for a project that uses Long Short-Term Memory (LSTM) models to predict the future closing price of stocks given historical data. It is a comprehensive and end-to-end machine learning project covering the whole process from data collection to model evaluation and prediction.

The stocks that we predict in this project include BlackRock, Goldman Sachs, JPMorgan Chase, and D. E. Shaw Group.

## Project Overview

1. **Data Collection:** 

   The project uses the Yahoo Finance API to download the historical stock data.

2. **Data Preprocessing:** 

   The data preprocessing includes scaling the closing prices of the stocks between 0 and 1, and preparing the data for the LSTM model.

3. **Model Building:** 

   The LSTM model consists of two LSTM layers each followed by a dropout layer to prevent overfitting. This is followed by two Dense layers.

4. **Model Training:** 

   The LSTM model is trained using the Adam optimizer and Mean Squared Error (MSE) as the loss function.

5. **Model Evaluation:** 

   The performance of the model is evaluated using the Mean Squared Error (MSE) between the actual and predicted stock prices.

6. **Prediction:**

   The LSTM model is used to predict the future closing prices of the stocks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project is implemented in Python. The required Python packages are listed in `requirements.txt`. You can install these with pip by running:

## Built With
Python 3 - The programming language used.
Keras - For building the LSTM model.
yfinance - Used to download historical stock data.
scikit-learn - For data preprocessing and evaluation.

## Graphs
1. **BR**
   ![image](https://github.com/DhruvAjayToshniwal/Stock-Price-Prediction-Using-LSTM/assets/57616258/82a22a74-cd0f-4cf6-a4df-543d06844397)

2. **GS**
   ![image](https://github.com/DhruvAjayToshniwal/Stock-Price-Prediction-Using-LSTM/assets/57616258/fe23593b-9ca4-43be-b8e1-7fc6e7a383ae)

3. **JPM**
   ![image](https://github.com/DhruvAjayToshniwal/Stock-Price-Prediction-Using-LSTM/assets/57616258/f2b99e0d-0563-49a9-8b11-75c12691663e)

4. **DE**
   ![image](https://github.com/DhruvAjayToshniwal/Stock-Price-Prediction-Using-LSTM/assets/57616258/d4558976-46b3-4bcf-991b-3b54e668c525)

```bash
pip install -r requirements.txt
