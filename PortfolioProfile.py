import pandas as pd
from ReturnProfile import asset_performance
from datetime import timedelta, date


class portfolio_performance:

    def __init__(self):
        
        self.lst_cls_securities = list()
        self.df_prices = pd.DataFrame()

        # //TODO: to get dates from excel
        date_from = date(2024, 1, 1)
        date_to = date(2024, 5, 17)

        # create a list of business days (w/o holidays)
        lst_bdays = pd.bdate_range(date_from, date_to)
        lst_bdays = [bday.date() for bday in lst_bdays]
        lst_bdays = [bday.isoformat() for bday in lst_bdays]

        # create an empty dataframe with business days as index
        self.df_prices = pd.DataFrame(lst_bdays)
        self.df_prices = self.df_prices.set_index(0)
        self.df_prices.index.name = 'date'


    def add_security(self,security):

        self.lst_cls_securities.append(security)


    def prepare_price_matrix(self):

        for sec in self.lst_cls_securities:

            df_temp = sec.df_analysis['close']
            df_temp.name = sec.ticker

            self.df_prices = pd.merge(self.df_prices,
                                    df_temp,
                                    how='left',
                                    left_index=True,
                                    right_index=True
                                    )
        
        self.df_prices.to_excel('temp/dirty_prices.xlsx')

    def clean_price_matrix(self):

        # analyse holidays //TODO: visualize the holidays and export into excel
        holidays_days = self.df_prices.isna().sum(axis=1)
        holidays_days = holidays_days[holidays_days != 0]

        holidays_tickers = self.df_prices.isna().sum(axis=0)

        # remove the holidays that are applicable to all the tickers (aka weekends)
        self.df_prices = self.df_prices[self.df_prices.count(axis=1) != 0]

        self.df_prices.fillna(method='ffill', inplace=True)
        self.df_prices.to_excel('temp/clean_prices.xlsx')


cls_asset1 = asset_performance(ticker='NVDA', b_from_FMP=True)
cls_asset2 = asset_performance(ticker='LHA.DE', b_from_FMP=True)

cls_portfolio = portfolio_performance()
cls_portfolio.add_security(cls_asset1)
cls_portfolio.add_security(cls_asset2)
cls_portfolio.prepare_price_matrix()
cls_portfolio.clean_price_matrix()


print(cls_portfolio.df_prices)
