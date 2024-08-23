import requests
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
import csv


# API bridge
def request_FMP_data(ticker, SDate, EDate):
    
    api_key = '1ZxcVL8gBZIFARcf1f9lOR4SZsK1lTSF'
    data_type = 'historical-price-full'
    url = f"https://financialmodelingprep.com/api/v3/{data_type}/{ticker}?from={EDate}&to={SDate}&apikey={api_key}"
    print(url)
    r = requests.get(url)
    #self.logger.info(f'Sending the following request {url}')
    return r.json()

def request_financials(ticker, fs_type, frequency='annual'):
    
    api_key = '1ZxcVL8gBZIFARcf1f9lOR4SZsK1lTSF'
    
    if fs_type == 'IS':
        data_type = 'income-statement'
    if fs_type == 'BS':
        data_type = 'balance-sheet-statement'
    if fs_type == 'CF':
        data_type = 'cash-flow-statement'

    url = f"https://financialmodelingprep.com/api/v3/{data_type}/{ticker}?period={frequency}&apikey={api_key}"
    print(url)
    
    r = requests.get(url)
    #self.logger.info(f'Sending the following request {url}')
    return r.json()

# Function to check if FMP json output is reliable
def request_is_good(json_r):
    
    b_return = False
    req_keys = list(json_r.keys())
    
    if req_keys == ['symbol', 'historical']:
        b_return = True

    return b_return


class FMP_download:


    def __init__(self, ticker, **kwargs):
        
        self.ticker = ticker
        self.SDate = kwargs.get('SDate','2024-07-01')
        self.EDate = kwargs.get('EDate','2024-01-01')
        self.df_output = pd.DataFrame()
        #self.market_data()


    def run_bulk_request(self):
        
        # API bridge
        def request_FMP_data(ticker, SDate, EDate):

            api_key = '1ZxcVL8gBZIFARcf1f9lOR4SZsK1lTSF'
            data_type = 'historical-price-full'
            url = f"https://financialmodelingprep.com/api/v3/{data_type}/{ticker}?from={EDate}&to={SDate}&apikey={api_key}"
            r = requests.get(url)
            #self.logger.info(f'Sending the following request {url}')
            return r.json()

        # Function to check if FMP json output is reliable
        def request_is_good(json_r):
            b_return = False
            req_keys = list(json_r.keys())
            if req_keys == ['symbol', 'historical']:
                b_return = True

            return b_return

    def market_data(self):

        data = request_FMP_data(self.ticker, self.SDate, self.EDate)
        self.data = data

        if request_is_good(data):

            df_hist_data = pd.json_normalize(data['historical'])

            str_csv_name = self.ticker + '.csv'
            str_path = 'market data/'

            df_hist_data.to_csv(str_path + str_csv_name, index=False)
            self.df_output = df_hist_data

        else:

            print(f'Could not download the market data for the ticker: {self.ticker}.')

    def financial_statement(self,fs_type:str):

        data = pd.json_normalize(request_financials(self.ticker,fs_type=fs_type))
        #self.data = pd.json_normalize(data)

        return data





# cls_request = FMP_download('ADBE')
# print(cls_request.df_output)
