import pandas as pd
import DownloadFMP
from DownloadFMP import FMP_download
from ReturnProfile import asset_performance
from PortfolioProfile import portfolio_performance

df_tickers = pd.read_csv('Tickers.csv')

cls_portfolio = portfolio_performance()

for ind in df_tickers.index:
    
    ticker = df_tickers.iloc[ind].values[0]
    cls_asset = asset_performance(ticker=ticker, b_from_FMP=True)
    cls_portfolio.add_security(cls_asset)

    del cls_asset

cls_portfolio.clean_price_matrix()
cls_portfolio.correlation_matrix()

