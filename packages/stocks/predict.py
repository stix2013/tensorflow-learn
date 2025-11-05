import yfinance as yf
from tensorflow.keras.models import load_model
import pickle

from parameters import (
    TICKER,
    TIME_STEP,
    MODEL_FILE,
    SCALER_FILE,
    TICKER_PERIOD,
    NUM_FUTURE_DAYS
)
from predict_multi_days import predict_multi_day

# --- Configuration ---
# TICKER = "GOOGL"
# START_DATE = "2023-01-01"
# END_DATE = "2025-11-03"
# TIME_STEP = 60 # Look-back period
# MODEL_FILE = 'GOOGLE_prediction_model.keras'
# SCALER_FILE = 'scaler.pkl'

loaded_model = load_model(MODEL_FILE)
print(f"✅ Model successfully loaded for prediction.")

# 1. Fetch the absolute most recent data (e.g., 90 days to ensure enough data for the 60-day window)
recent_df = yf.download(TICKER, period=TICKER_PERIOD)
recent_prices = recent_df['Close'].values.reshape(-1, 1)

# 2. Extract the last window needed for the prediction
last_window = recent_prices[-TIME_STEP:]

# 3. Load the scaler
with open(SCALER_FILE, 'rb') as file:
    loaded_scaler = pickle.load(file)

# Scaled the prices data
last_window_scaled = loaded_scaler.transform(last_window)

# 4. Scale and Reshape the window
# scaled_input = loaded_scaler.transform(last_window)
# X_next_day = scaled_input.reshape(1, TIME_STEP, 1)

# 5. Make the Prediction
# scaled_prediction = loaded_model.predict(X_next_day)

# 6. Inverse Transform to get the actual price
# predicted_price = loaded_scaler.inverse_transform(scaled_prediction)

# print(f"\n--- Next Day Prediction ---")
# print(f"Date of last data point: {recent_df.index[-1].strftime('%Y-%m-%d')}")
# print(f"Predicted closing price for the next market day: $ {predicted_price[0, 0]:.2f}")

future_forecast = predict_multi_day(loaded_model, loaded_scaler, last_window_scaled, NUM_FUTURE_DAYS, TIME_STEP)

print(f"\n{NUM_FUTURE_DAYS}-Day Forecast:\n{future_forecast}")