from keras.models import load_model
import numpy as np
import pickle

# Load the data and scalers from pickle files
with open('processed_data/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

with open('processed_data/train_test_data.pkl', 'rb') as f:
    train_test_data = pickle.load(f)

# Define list of stocks
stocks = ['BLK', 'GS', 'JPM', 'DE']

# Loop over stocks again to make future predictions
for stock in stocks:
    print(f"Predicting next day's closing price for stock: {stock}")

    # Load the trained model from previous step
    model = load_model(f'trained_models/{stock}_model.h5')

    # Load the train-test data and scaler for the stock
    x_train, x_test, y_train, y_test = train_test_data[stock]
    scaler = scalers[stock]

    # Get the last element of the entire dataset (train + test)
    x_input = np.concatenate((x_train, x_test))[-1].reshape((1, 1, 1))

    # Predict the next day's closing price
    pred_price = model.predict(x_input)

    # Reverse the scaling of the prediction
    pred_price = scaler.inverse_transform(pred_price)

    print(f"Predicted closing price for {stock} on next day: {pred_price[0][0]}")
