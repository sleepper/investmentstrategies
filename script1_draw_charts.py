import pandas as pd
import DownloadFMP
from DownloadFMP import FMP_download
from ReturnProfile import asset_performance

df_tickers = pd.read_csv('Tickers.csv')

# smth is not working, when looped
for ind in df_tickers.index:
    
    ticker = df_tickers.iloc[ind].values[0]
    cls_asset = asset_performance(ticker=ticker, b_from_FMP=True)
    del cls_asset


