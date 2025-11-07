from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError

def create_model(time_step, lstm_unit_1 = 64, lstm_unit_2 =64, dense_units = 128, dropout_ratio = 0.8):
    model = Sequential([
        # First Layer
        LSTM(units=lstm_unit_1, return_sequences=True, input_shape=(time_step,1)),

        # Second Layer
        LSTM(units=lstm_unit_2, return_sequences=False),

        # 3rd Layer
        Dense(units=dense_units, activation="relu"),

        # 4th Layer
        Dropout(dropout_ratio),

        # Output layer
        Dense(units=1)
    ])

    # Compile the Model
    model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=[RootMeanSquaredError()]
        )
    
    print("Model created and compiled.")

    return model