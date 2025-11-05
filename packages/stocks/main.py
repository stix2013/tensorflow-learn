import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
import pickle

from parameters import (
    TICKER,
    TIME_STEP,
    DENSE_NEURONS,
    DROPOUT_NEURONS,
    TRAIN_DATA_PERC,
    MODEL_FILE,
    SCALER_FILE
)

from create_model import create_model

# --- Configuration ---
# TICKER = "GOOGL"
START_DATE = "2022-01-01"
END_DATE = "2025-11-04"

# 1. Retrieve Data
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# 2. Prepare Data for Scaling (Use 'Close' prices)
prices = df['Close'].values.reshape(-1, 1)

# 3. Initialize and Fit Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# 4. Save the fitted scaler (CRITICAL for prediction)
with open(SCALER_FILE, 'wb') as file:
    pickle.dump(scaler, file)
print(f"✅ Scaler saved to {SCALER_FILE}")

# 5. Create Time-Step Sequences (X: input, y: target)
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, TIME_STEP)

# 6. Reshape for LSTM: [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 7. Split Data (e.g., 80% train, 20% test)
train_size = int(len(X) * TRAIN_DATA_PERC)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Define the Keras Model
model = create_model(time_step=TIME_STEP, dense_units=DENSE_NEURONS, dropout_ratio=DROPOUT_NEURONS)

# 3. Train the Model
# Note: Training on the validation set (X_test, y_test) is common for checking performance
# but for a final model, you might train on all data or use a separate validation split.
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=64,
    epochs=100, # Adjust epochs based on complexity and time
    verbose=1
)
print("Model training finished.")

model.save(MODEL_FILE)
print(f"✅ Trained model saved to {MODEL_FILE}")
