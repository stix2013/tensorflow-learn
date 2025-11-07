import yfinance as yf
import pandas as pd

# 1. Define the ticker and date range
ticker_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-10-31"

# 2. Download the data
data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=False)

# 'data' is now a pandas DataFrame containing 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume'
# remove 3rd first row
print(data.head()['Volume'])
data.to_csv('AppleTicker.csv')