import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle

# Define function to prepare data for LSTM model
def create_lstm_dataset(data, look_back=1):
    x_data, y_data = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        x_data.append(a)
        y_data.append(data[i + look_back, 0])
    return np.array(x_data), np.array(y_data)

# Define list of stocks
stocks = ['BLK', 'GS', 'JPM', 'DE']

# Create directories for storing models and processed data
os.makedirs('trained_models', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

# Dictionaries to store scalers and train_test_data
scalers = {}
train_test_data = {}

# Loop over stocks
for stock in stocks:
    print(f"Processing stock: {stock}")
    
    # Load data
    try:
        data = yf.download(stock, start='2020-01-01', end='2022-12-31')
    except Exception as e:
        print(f"Error downloading data for stock {stock}: {str(e)}")
        continue

    # Preprocessing
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    
    # Prepare LSTM training set
    look_back = 1
    x_data, y_data = create_lstm_dataset(scaled_data, look_back)
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Reshape input to be [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    scalers[stock] = scaler  # Store the scaler
    train_test_data[stock] = (x_train, x_test, y_train, y_test)  # Store the train-test data

# Saving the data and scalers using pickle
with open('processed_data/scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
    
with open('processed_data/train_test_data.pkl', 'wb') as f:
    pickle.dump(train_test_data, f)
