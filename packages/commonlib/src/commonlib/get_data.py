from pandas import read_csv
from yfinance import download
from datetime import datetime
from dateutil.relativedelta import relativedelta
from os import path

def get_data(ticker_symbol = 'AAPL', year_back=1, interval='1d', saved_filename = 'data.csv', used_saved=True):
  now = datetime.now()
  end_date = now.strftime('%Y-%m-%d')
  start_date = (now - relativedelta(years=year_back)).strftime('%Y-%m-%d')

  if path.isfile(saved_filename) and not used_saved:
    data = read_csv(saved_filename)
    data = data.iloc[2:]
  else:
    data = download(ticker_symbol, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    data.to_csv(saved_filename)

  return data
