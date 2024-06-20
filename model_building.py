from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pickle
import os

# Create directory for storing models
os.makedirs('trained_models', exist_ok=True)

# Loading the data from pickle file
with open('processed_data/train_test_data.pkl', 'rb') as f:
    train_test_data = pickle.load(f)

# Define list of stocks
stocks = ['BLK', 'GS', 'JPM', 'DE']

# Loop over stocks
for stock in stocks:
    print(f"Building model for stock: {stock}")

    # Load the train data
    x_train, _, y_train, _ = train_test_data[stock]

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    try:
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        model_path = f'trained_models/{stock}_model.h5'
        model.save(model_path)  # Save the model to a file
    except Exception as e:
        print(f"Error training model for stock {stock}: {str(e)}")
