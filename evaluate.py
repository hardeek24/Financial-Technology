import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load the data and scalers from pickle files
with open('processed_data/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

with open('processed_data/train_test_data.pkl', 'rb') as f:
    train_test_data = pickle.load(f)

# Define list of stocks
stocks = ['BLK', 'GS', 'JPM', 'DE']

# Loop over stocks again to evaluate the models
for stock in stocks:
    print(f"Evaluating model for stock: {stock}")

    # Load the trained model from previous step
    model = load_model(f'trained_models/{stock}_model.h5')

    # Load the train-test data and scaler for the stock
    _, x_test, _, y_test = train_test_data[stock]
    scaler = scalers[stock]

    # Get the last element of the training set
    x_input = x_test[0].reshape((1, 1, 1))

    # Create a list to hold the model's predictions
    predictions = []

    # Predict the closing price for the next day for each day in the test set
    for _ in range(len(x_test)):
        pred_price = model.predict(x_input)
        predictions.append(pred_price)

        # Update x_input to include the prediction and drop the oldest price
        x_input = np.append(x_input[:,1:,:], pred_price).reshape((1, 1, 1))

    # Reverse the scaling of the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Calculate mean squared error of the predictions
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error for {stock}: {mse}")

    # Plot the actual vs predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(range(len(predictions)), predictions, color='red', label='Predicted Price')
    plt.plot(range(len(predictions)), scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Price')
    plt.title(f'{stock} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
