import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from data_loader import load_market_data

############### IMPORT DATA #####################################
#portfolio tickers
tickers=['AAPL', 'MSFT', 'NVDA', 'GOOG']
# print('Downloading market data...')
# data=yf.download(tickers, start='2020-01-01', auto_adjust=True)
# print(data.head())
# prices=data['Close']
# prices.to_pickle('data/stock_prices.pkl')
# prices.to_csv('stock_prices.csv')
#prices=pd.read_csv('stock_prices.csv',index_col=0, parse_dates=True)
prices=load_market_data()
returns=prices.pct_change()
print('\n Daily Returns')
print(returns.head())

weights=[0.25,0.25,0.25,0.25]
portfolio_return=(returns*weights).sum(axis=1)
print(portfolio_return.head())

# annual_return=((1+portfolio_return.mean())**252-1)
# print(f'Annual Return: {annual_return: .2%}')
##########ANNUAL RETURNS##############
stock_annual_returns=(1+returns).groupby(returns.index.year).prod()-1
stock_annual_returns_pct=stock_annual_returns*100
print("stock annual returns: ", stock_annual_returns_pct.round(2))
portfolio_annual_return=(1+portfolio_return).groupby(portfolio_return.index.year).prod()-1
portfolio_annual_return_pct=portfolio_annual_return*100
print("portfolio_annual_returns: ", portfolio_annual_return_pct.round(2))

################ CAGR #################################
n_years=len(portfolio_return)/252
cumulative_growth=(1+portfolio_return).prod()
cagr=(cumulative_growth**(1/n_years)-1)
print(f'CAGR during 2021 till today is: {cagr:.2%}')

stock_cagr=(prices.iloc[-1]/prices.iloc[0])**(1/n_years)-1

############## VOLATILITY #########################
year_to_year_volatility = portfolio_annual_return.std()
print(f"\n Portfolio Year_to_Year Return Standard Deviation: {year_to_year_volatility:.2%}")

port_daily_volatility=portfolio_return.std()
annualized_daily_volatility=(port_daily_volatility*np.sqrt(252))
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

##################### SKEWNESS KURTOSIS #############################################
returns=prices.pct_change().dropna()

for stock in returns.columns:
    skew_value=skew(returns[stock])
    kurt_value=kurtosis(returns[stock])
    print(f'{stock}')
    print(f'Skewness: {skew_value: .2f}')
    print(f'Kurtosis: {kurt_value: .2f}')
    print('-'*40)
port_skew=skew(portfolio_return)
port_kurt=kurtosis(portfolio_return)
print(f'Portfolio Skewness: {port_skew: .2f}')
print(f'Portfolio Kurtosis: {port_kurt: .2f}')
print('-'*40)

risk_shape=pd.DataFrame({'Annual Returns': stock_cagr, 'Volatility': returns.std()*np.sqrt(252), 'Skewness': returns.skew(), 'Kurtosis': returns.kurt()})
risk_shape.loc['Portfolio']=[cagr,annualized_daily_volatility,port_skew, port_kurt]
report_risk_shape=pd.DataFrame(index=risk_shape.index)
report_risk_shape['Annual Returns']=risk_shape['Annual Returns'].map('{:.2%}'.format)
report_risk_shape['Volatility']=risk_shape['Volatility'].map('{:.2%}'.format)
report_risk_shape['Skewness']=risk_shape['Skewness'].map('{:.2f}'.format)
report_risk_shape['Kurtosis']=risk_shape['Kurtosis'].map('{:.2f}'.format)
print(report_risk_shape.sort_values(by='Kurtosis',ascending=False))

plt.figure(figsize=(10,5))
plt.hist(returns['MSFT'], bins=50, alpha=0.5, label='MSFT')
plt.hist(returns['GOOG'], bins=50, alpha=0.5, label='GOOG')
plt.legend()
plt.title('Return Distribution')
plt.show()

###################### VaR Expected Shortfall ######################################
var_HS_95_port=portfolio_return.quantile(0.05)
es_95_port=portfolio_return[portfolio_return<var_HS_95_port].mean()
var_HS_99_port=portfolio_return.quantile(0.01)
es_99_port=portfolio_return[portfolio_return<var_HS_99_port].mean()
print(f'95% VaR of HS: {var_HS_95_port: .2%}')
print(f'99% VaR of HS: {var_HS_99_port: .2%}')
var_HS_95_10d_port=var_HS_95_port*np.sqrt(10)
var_HS_99_10d_port=var_HS_99_port*np.sqrt(10)

var_HS_ES=pd.DataFrame({'HS_VaR_95%': returns.quantile(0.05), 'ES_95%': returns[returns<returns.quantile(0.05)].mean(), 'HS_VaR_99%': returns.quantile(0.01), 'ES_99%': returns[returns<returns.quantile(0.01)].mean()})
var_HS_ES.loc['Portfolio']=[var_HS_95_port, es_95_port, var_HS_99_port, es_99_port]
report_var_HS_ES=pd.DataFrame(index=var_HS_ES.index)
report_var_HS_ES['HS_VaR_95%']=var_HS_ES['HS_VaR_95%'].map('{:.2%}'.format)
report_var_HS_ES['ES_95%']=var_HS_ES['ES_95%'].map('{:.2%}'.format)
report_var_HS_ES['HS_VaR_99%']=var_HS_ES['HS_VaR_99%'].map('{:.2%}'.format)
report_var_HS_ES['ES_99%']=var_HS_ES['ES_99%'].map('{:.2%}'.format)
print('VaR - HS - ES')
print(report_var_HS_ES)

# plt.hist(portfolio_return, bins=50)
# plt.axvline(var_HS_95,color='red')
# plt.show()
risk_report=pd.concat([report_risk_shape, report_var_HS_ES], axis=1)
print(risk_report)

plt.hist(portfolio_return, bins=50)
plt.axvline(var_HS_95_port, color='red', label='VaR_HS_95')
plt.axvline(es_95_port, color='purple', label='ES95')
plt.show()

################## VC VaR ################################################
z_95=1.645
z_99=2.326
stock_daily_volatility=returns.std()
VC_VaR_95=-z_95*stock_daily_volatility
VC_VaR_99=-z_99*stock_daily_volatility
VC_VaR_95_port=-z_95*port_daily_volatility
VC_VaR_99_port=-z_99*port_daily_volatility
VC_VaR_all=pd.DataFrame({'VC_VaR_95%': VC_VaR_95, 'VC_VaR_99%': VC_VaR_99})
VC_VaR_all.loc['Portfolio']=[VC_VaR_95_port, VC_VaR_99_port]
report_VC_VaR=pd.DataFrame(index=VC_VaR_all.index)
report_VC_VaR['VC_VaR_95%']=VC_VaR_all['VC_VaR_95%'].map('{:.2%}'.format)
report_VC_VaR['VC_VaR_99%']=VC_VaR_all['VC_VaR_99%'].map('{:.2%}'.format)
print(report_VC_VaR)

risk_report=pd.concat([risk_report,report_VC_VaR], axis=1)
risk_report=risk_report[['Annual Returns', 'Volatility', 'Skewness', 'Kurtosis', 'VC_VaR_95%', 'HS_VaR_95%', 'ES_95%', 'VC_VaR_99%', 'HS_VaR_99%', 'ES_99%']]
print(risk_report)
################## RISK ATTRIBUTION ########################################
cov_matrix=returns.cov()
mrc=cov_matrix@weights/port_daily_volatility
rc=weights*mrc
rc_pct=rc/rc.sum()
print("Risk Contribution: ")
print(rc_pct.map('{:.2%}'.format))

###################### EFFICIENCY #########################################
return_contribution=stock_cagr*weights
return_contribution_pct=return_contribution/return_contribution.sum()
efficiency=return_contribution_pct/rc_pct
report_attribution=pd.DataFrame({'Weight': weights, 'Return Contribution': return_contribution_pct, 'Risk Contribution': rc_pct, 'Efficiency': efficiency}, index=returns.columns)
report_attribution['Weight']=report_attribution['Weight'].map('{:.2%}'.format)
report_attribution['Return Contribution']=report_attribution['Return Contribution'].map('{:.2%}'.format)
report_attribution['Risk Contribution']=report_attribution['Risk Contribution'].map('{:.2%}'.format)
report_attribution['Efficiency']=report_attribution['Efficiency'].map('{:.2f}'.format)
print(report_attribution)