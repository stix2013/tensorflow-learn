from .create_model import create_model
from .get_data import get_data
from .predict_multi_days import predict_multi_day

# --- Parameters ---
TICKER = 'GOOGL'
TICKER_PERIOD = '90d'
YEAR_COUNT = 2
START_DATE = '2020-01-01'
END_DATE = '2025-11-03'
TIME_STEP = 60 # Look-back period
MODEL_FILE = 'GOOGLE_prediction_model.keras'
CSV_FILENAME = 'data.csv'
SCALER_FILE = 'scaler.pkl'
TIME_STEP = 30
ATE_COL_NAME = 'Date'
PRICE_COL_NAME = 'Close'
WINDOW_SIZE = 60
SAVE_MODEL = False
DATE_USED_SAVED = True
BATCH_SIZE = 32
TRAIN_DATA_PERC = 0.9
NUM_FUTURE_DAYS = 3

EPOCH_STEPS = 100
# Layers units
LSTM_UNIT_1 = 64
LSTM_UNIT_2 = 64
DENSE_NEURONS = 128
DENSE_ACT_METHODS = 'relu'
DROPOUT_RATE = 0.3
OUTPUT_NEURON = 1

__all__ = [
    create_model,
    get_data,
    predict_multi_day
]