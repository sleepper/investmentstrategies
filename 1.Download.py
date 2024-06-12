import requests
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
import csv

logger = logging.getLogger(__name__)

# Setting up a logger
def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('log.log'),
            logging.StreamHandler()
        ]
    )

if __name__ == '__main__':
    main()

# API bridge
def request_FMP_data(ticker):

    api_key = '1ZxcVL8gBZIFARcf1f9lOR4SZsK1lTSF'
    data_type = 'historical-price-full'
    date_from = '2024-01-01'
    date_to = '2024-05-17'
    url = f"https://financialmodelingprep.com/api/v3/{data_type}/{ticker}?from={date_from}&to={date_to}&apikey={api_key}"
    r = requests.get(url)
    logger.info(f'Sending the following request {url}')
    return r.json()

# Function to check if FMP json output is reliable
def request_is_good(json_r):
    b_return = False
    req_keys = list(json_r.keys())
    if req_keys == ['symbol', 'historical']:
        b_return = True

    return b_return

# Read the tickers, saved in a separate csv file
logger.info(f'Reading the tickers from the current protfolio')
df_tickers = pd.read_csv('Tickers.csv')

# Loop through tickers, download the market data and list the errors
lst_ticker = list(df_tickers['Tickers'])
lst_problematic_tickers = []

for ticker in lst_ticker:

    data = request_FMP_data(ticker)
    logger.info(f'Requesting the market data for {ticker}')

    if request_is_good(data):

        logger.info('Received data for ' + ticker)

        df_hist_data = pd.DataFrame.from_dict(
            data['historical'][0], orient='index').T

        n_rows = len(data['historical'])

        for i in range(1, n_rows):

            df_temp = pd.DataFrame.from_dict(
                data['historical'][i], orient='index').T
            df_hist_data = df_hist_data.append(df_temp)

        str_csv_name = ticker + '.csv'
        str_path = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\market data\\'

        df_hist_data.to_csv(str_path + str_csv_name, index=False)
        logger.info('Market data for ' + ticker + ' saved')

    else:

        lst_problematic_tickers.append(ticker)
        logger.info('Error while processing ' + ticker)

        df_output = pd.DataFrame(lst_problematic_tickers)
        df_output.to_csv('Problematic tickers.csv', sep=',', index=False)
