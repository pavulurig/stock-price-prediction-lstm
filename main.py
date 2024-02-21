# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
file_path = 'TSLA.csv'  # Path to the Tesla stock price CSV file
df = pd.read_csv(file_path)

# We use only the 'Close' price as our feature for simplicity
data = df[['Close']].values

# Normalize the 'Close' prices using MinMaxScaler to fit the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Define a function to create sequences from the dataset
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  # Input data (features)
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])  # Output data (target)
    return np.array(dataX), np.array(dataY)

# Split the data into training and test sets
training_size = int(len(data_scaled) * 0.67)  # 67% of data for training
test_size = len(data_scaled) - training_size  # Remaining data for testing
train_data, test_data = data_scaled[0:training_size, :], data_scaled[training_size:len(data_scaled), :]

# Generate training and testing data with a specified time step
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM models
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(100, 1)),  # First LSTM layer
    tf.keras.layers.LSTM(50, return_sequences=False),  # Second LSTM layer
    tf.keras.layers.Dense(25),  # Dense layer with 25 neurons
    tf.keras.layers.Dense(1)  # Output layer with 1 neuron (prediction)
])

# Compile the model specifying the optimizer and loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with the training data
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Make predictions on the training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate and print RMSE for both training and test data
train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Plot the actual prices and the model's predictions
plt.figure(figsize=(15, 6))
plt.plot(df['Close'].values, label='Actual Price')  # Actual prices
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict.ravel(), label='Training Predictions')  # Training predictions
plt.plot(np.arange(len(train_predict) + (time_step*2) + 1, len(train_predict) + (time_step*2) + 1 + len(test_predict)), test_predict.ravel(), label='Test Predictions')  # Test predictions
plt.legend()
plt.show()
