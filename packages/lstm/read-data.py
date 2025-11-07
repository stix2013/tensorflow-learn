from pandas import read_csv

data = read_csv('AppleTicker.csv')
test_data = data[2:].copy
print(test_data.head())
print(test_data.info())
print(test_data.describe())