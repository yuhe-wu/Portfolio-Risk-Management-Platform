import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############### IMPORT DATA #####################################
#portfolio tickers
tickers=['AAPL', 'MSFT', 'NVDA', 'GOOG']
print('Downloading market data...')
data=yf.download(tickers, start='2020-01-01', auto_adjust=True)
print(data.head())
prices=data['Close']
returns=prices.pct_change()
print('\n Daily Returns')
print(returns.head())

weights=[0.25,0.25,0.25,0.25]
portfolio_return=(returns*weights).sum(axis=1)
print(portfolio_return.head())

# annual_return=((1+portfolio_return.mean())**252-1)
# print(f'Annual Return: {annual_return: .2%}')
##########ANNUAL RETURNS##############
n_years=len(portfolio_return)/252
cumulative_growth=(1+portfolio_return).prod()
cagr=(cumulative_growth**(1/n_years)-1)
print(f'CAGR during 2021 till today is: {cagr:.2%}')

annual_returns=(1+returns).groupby(returns.index.year).prod()-1
annual_returns_pct=annual_returns*100
print("stock annual returns: ", annual_returns_pct.round(2))
portfolio_annual_returns=(1+portfolio_return).groupby(portfolio_return.index.year).prod()-1
portfolio_annual_returns_pct=portfolio_annual_returns*100
print("portfolio_annual_returns: ", portfolio_annual_returns_pct.round(2))
############## VOLATILITY #########################
year_to_year_volatility = portfolio_annual_returns.std()
print(f"\n Portfolio Year_to_Year Return Standard Deviation: {year_to_year_volatility:.2%}")

daily_volatility=portfolio_return.std()
annualized_daily_volatility=(daily_volatility*np.sqrt(252))
print(f"\nAnnualized Daily Volatility: {annualized_daily_volatility: .2%}")

############### SHARPE RATIO ##############################
sharpe_ratio=cagr/annualized_daily_volatility
print(f"Sharpe Ratio: {sharpe_ratio: .2f}")

############## WEALTH INDEX ###################################
wealth_index = (1+portfolio_return).cumprod()
print("Wealth Index Header: ", wealth_index.head())

############## MAX DRAWDOWN ###################################
running_max=wealth_index.cummax()
drawdown=(wealth_index/running_max-1)
max_drawdown=drawdown.min()
print(f"Max Drawdown: {max_drawdown: .2%}")

############# GRAPHING  #######################################
wealth_index.plot(title="Portfolio Growth")
plt.show()
drawdown.plot(title="DrawDown")
plt.show()

################## PORTFOLIO RISK SUMMARY ##########################
print('\n ========================================================================================')
print("Portfolio Analytics")
print('=========================================================================================')
print (f'CAGR: {cagr: .2%}')
print(f'Volatility: {annualized_daily_volatility: .2%}')
print(f'Sharpe Ratio: {sharpe_ratio: .2f}')
print(f'Max Drawdown: {max_drawdown: .2%}')

################# MARKET CONTEXT ####################################
for ticker in tickers:
    price_series=prices[ticker]
    recent=price_series.tail(252)
    current_price=recent.iloc[-1]
    high_52w=recent.max()
    low_52w=recent.min()
    high_date=recent.idxmax()
    low_date=recent.idxmin()
    position_in_range=(current_price-low_52w)/(high_52w-low_52w)
    distance_from_high=(current_price/high_52w)-1
    distance_from_low=(current_price/low_52w)-1
    percentile=((recent<current_price).mean())
    mean_price=recent.mean()
    std_price=recent.std()
    z_score=(current_price-mean_price)/std_price
    print(f'\n {ticker}')
    print(f'Current Price: {current_price: .2f}')
    print(f'52W High: {high_52w: .2f}' f'({high_date.date()})')
    print(f'52w Low: {low_52w: .2f}' f'({low_date.date()})')
    print(f'Position in Range: ' f'{position_in_range: .2%}')
    print(f'Distance from High: ' f'{distance_from_high: .2%}')
    print(f'Distance from Low: ' f'{distance_from_low: .2%}')
    print(f'Price Percentile: {percentile: .2%}')
    print(f'Z-Score: {z_score: .2f}')

