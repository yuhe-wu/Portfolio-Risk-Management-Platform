import pandas as pd
import os
import yfinance as yf

tickers=['AAPL', 'MSFT', 'NVDA', 'GOOG']

def load_market_data():
    file_path='data/stock_prices.pkl'
    if os.path.exists(file_path):
        print('loading data from local')
        return pd.read_pickle(file_path)
    else:
        print('Downloading data ...')
        data=yf.download(tickers, start='2020-01-01', auto_adjust=True)
        prices=data['Close']
        prices.to_pickle(file_path)
        return prices

price=load_market_data()
print(price.head())