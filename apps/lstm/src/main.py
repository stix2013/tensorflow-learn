from tensorflow import keras
from tensorflow import config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime
from py_commonlib.get_data import get_data

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CSV_FILENAME = "AAPL_stocks.csv"
SCALER_FILENAME = "AAP_scaler.pkl"
DATE_COL_NAME = "Date"
PRICE_COL_NAME = "Adj Close"
TRAINING_RATE = 0.6
WINDOW_SIZE = 60
SAVE_MODEL = False
EPOCH_STEPS = 100
BATCH_SIZE = 32
DROPOUT_NEURONS = 0.3
OUTPUT_NEURON = 1
UNIT_LSTM_1 = 64
UNIT_LSTM_2 = 64
DENSE_NEURONS = 128
DENSE_ACT_METHODS = "relu"

print("Get data from CSV")
data = pd.read_csv(CSV_FILENAME)
# print("Get data from Yahoo Finance")
# data = get_data(used_saved=False)

# print(data.head())
# print(data.info())
# print(data.describe())

# Data Visualization
# plt.figure(figsize=(12,6))
# plt.plot(data['Date'], data['Close'],label="Close",color="blue")
# plt.plot(data['Date'], data['Open'],label="Open",color="red")
# plt.legend()
# plt.show()

# plt.figure(figsize=(12,6))
# plt.plot(data['Date'], data['Volume'],label="Volume",color="orange")
# plt.show()

# select only integer and float
# numeric_data = data.select_dtypes(include=["int64","float64"])
# plt.figure(figsize=(8,6))
# sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
# plt.show()

data[DATE_COL_NAME] = pd.to_datetime(data[DATE_COL_NAME])
print("Date: {data[DATE_COL_NAME]}")

prediction = data.loc[
    (data[DATE_COL_NAME] > datetime(2022,1,1)) &
    (data[DATE_COL_NAME] < datetime(2025,11,5))
]

# plt.figure(figsize=(12,6))
# plt.plot(data["Date"],data["Close"],color="green")
# plt.xlabel("Date")
# plt.ylabel("Close")

# Prepare
dataset = data.filter([PRICE_COL_NAME]).values
training_data_len = int(np.ceil(len(dataset) * TRAINING_RATE))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

# Save the fitted scaler (CRITICAL for prediction)
with open(SCALER_FILENAME, 'wb') as file:
    pickle.dump(scaler, file)
print(f"✅ Scaler saved to {SCALER_FILENAME}")

training_data = scaled_data[:training_data_len]

X_train, Y_train = [], []

# Create a sliding window for our stocks
for i in range(WINDOW_SIZE, len(training_data)):
    X_train.append(training_data[i-WINDOW_SIZE:i, 0])
    Y_train.append(training_data[i,0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# model
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(UNIT_LSTM_1, return_sequences=True, input_shape=(X_train.shape[1],1)))

# Second Layer
model.add(keras.layers.LSTM(UNIT_LSTM_2, return_sequences=False))

# 3rd Layer
model.add(keras.layers.Dense(DENSE_NEURONS, activation=DENSE_ACT_METHODS))

# 4th Layer
model.add(keras.layers.Dropout(DROPOUT_NEURONS))

# Output layer
model.add(keras.layers.Dense(OUTPUT_NEURON))

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mae",
    metrics=[keras.metrics.RootMeanSquaredError()]
)

training = model.fit(X_train, Y_train, epochs=EPOCH_STEPS, batch_size=BATCH_SIZE)

# Prep test data
test_data = scaled_data[training_data_len - WINDOW_SIZE:]
X_test, Y_test = [], dataset[training_data_len:]

for i in range(WINDOW_SIZE, len(test_data)):
    X_test.append(test_data[i-WINDOW_SIZE:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# print("X Test Data:")
# print(X_test)

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# plot
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

if SAVE_MODEL:
    model.save('AAPL_stocks_model.keras')

plt.figure(figsize=(12,8))
plt.plot(train[DATE_COL_NAME], train[PRICE_COL_NAME], label="Train (Actual)", color="blue")
plt.plot(test[DATE_COL_NAME], test[PRICE_COL_NAME], label="Test (Actual)", color="orange")
plt.plot(test[DATE_COL_NAME], test['Predictions'], label="Predictions", color="red")
plt.title("Our Stock Predictions")

plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
