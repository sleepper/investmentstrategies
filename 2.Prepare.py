import pandas as pd
import numpy as np
import os
from datetime import timedelta, date
import matplotlib.pyplot as plt

# //TODO: to get dates from excel
date_from = date(2024, 1, 1)
date_to = date(2024, 5, 17)

# directory with csv files
str_path = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\market data\\'
lst_available_tickers = os.listdir(str_path)

# create a list of business days (w/o holidays)
lst_bdays = pd.bdate_range(date_from, date_to)
lst_bdays = [bday.date() for bday in lst_bdays]
lst_bdays = [bday.isoformat() for bday in lst_bdays]

# create an empty dataframe with business days as index
df_market_data = pd.DataFrame(lst_bdays)
df_market_data = df_market_data.set_index(0)
df_market_data.index.name = 'Date'

# create a list with tickers, for which CSV is available
lst_available_tickers = [ticker.replace(
    '.csv', '') for ticker in lst_available_tickers]

# collect market data into one dataframe
for ticker in lst_available_tickers:

    df_temp = pd.read_csv(str_path+ticker+'.csv')
    df_temp = df_temp.set_index('Date')
    # there is more data, adjClose is a starter
    df_temp = df_temp[['adjClose']]
    df_temp.rename(columns={'adjClose': ticker}, inplace=True)

    df_market_data = pd.merge(df_market_data,
                              df_temp,
                              how='left',
                              left_index=True,
                              right_index=True
                              )

# analyse holidays //TODO: visualize the holidays and export into excel
holidays_days = df_market_data.isna().sum(axis=1)
holidays_days = holidays_days[holidays_days != 0]

holidays_tickers = df_market_data.isna().sum(axis=0)

# remove the holidays that are applicable to all the tickers (aka weekends)
df_market_data = df_market_data[df_market_data.count(axis=1) != 0]

# identify the dates //TODO: ideally the index should be sortable. Maybe it should be in datetime data format
# the navigation is also not straight forward with switching between index ids and index labels
period_end = date_to.isoformat()
# all close prices are available
check = sum(df_market_data.loc[[period_end]].isna().sum()) == 0

if check:
    print(f'The end date {period_end} is valid')
else:
    print(
        f'Select another end date, since not all prices are available for {period_end}')

period_offset = 252/4  # calendar quarter
# reverse for the existing order
period_offset = len(df_market_data) - period_offset

period_start = df_market_data.iloc[[period_offset]].index.tolist()[0]
check = sum(df_market_data.loc[[period_start]].isna().sum()) == 0

if check:
    print(f'The end date {period_start} is valid')
else:
    print(
        f'Select another end date, since not all prices are available for {period_start}')

# Calculating simple holding period price performance
period_returns = df_market_data.loc[period_end] / \
    df_market_data.loc[period_start]

# //TODO: add the dividends

# Computing inputs for mean-variance analysis
# replace NULL values with last available prices
df_market_data.fillna(method='bfill', inplace=True)

# df_absolute_returns = df_market_data.pct_change() - not really relevant
df_log_returns = df_market_data.shift(0).apply(
    np.log) - df_market_data.shift(1).apply(np.log)

# resize the dataframe for the analysis period
id_end = int(len(df_market_data))
id_start = int(id_end - period_offset)
df_log_returns = df_log_returns.iloc[id_start:id_end]

# perform the return and vola analysis
ser_avg_daily_log_returns = df_log_returns.mean()
ser_avg_daily_log_returns_ann = ser_avg_daily_log_returns.apply(
    lambda x: (1+x) ** 252 - 1)  # annualized

ser_avg_daily_vola = df_log_returns.std()
ser_avg_daily_vola_ann = ser_avg_daily_vola.apply(
    lambda x: x * np.sqrt(252))  # annualized

df_mean_var_output = pd.DataFrame(
    [df_log_returns.mean(), df_log_returns.std()], index=['Avg. return', 'Vola'])

# df_mean_var_output.T.plot.scatter(x=1,y=0)
