from tensorflow.keras.models import load_model

MODEL_FILENAME = "AAPL_stocks_model.keras"

model = load_model(MODEL_FILENAME)

print("Successfully loaded")
model.summary()
