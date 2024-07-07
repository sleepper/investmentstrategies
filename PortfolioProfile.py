import pandas as pd
from ReturnProfile import asset_performance
from datetime import timedelta, date
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import CopulaDistribution, GaussianCopula
import numpy as np


class portfolio_performance:

    def __init__(self):
        
        self.lst_cls_securities = list()
        self.df_prices = pd.DataFrame()

        # //TODO: to get dates from excel
        self.date_from = date(2024, 1, 1)
        self.date_to = date(2024, 5, 17)

        # create a list of business days (w/o holidays)
        self.lst_bdays = pd.bdate_range(self.date_from, self.date_to)
        self.lst_bdays = [bday.date() for bday in self.lst_bdays]
        self.lst_bdays = [bday.isoformat() for bday in self.lst_bdays]

        # # create an empty dataframe with business days as index
        # self.df_prices = pd.DataFrame(self.lst_bdays)
        # self.df_prices = self.df_prices.set_index(0)
        # self.df_prices.index.name = 'date'

        # add tickers

        self.lst_tickers = list()


    def add_security(self,security):

        self.lst_cls_securities.append(security)


    def __prepare_dirty_price_matrix__(self):

        # create an empty dataframe with business days as index
        self.df_prices = pd.DataFrame(self.lst_bdays)
        self.df_prices = self.df_prices.set_index(0)
        self.df_prices.index.name = 'date'
        
        # is a separate step for reconciliation purposes
        for sec in self.lst_cls_securities:

            df_temp = sec.df_analysis['close']
            df_temp.name = sec.ticker
            self.lst_tickers.append(sec.ticker)

            self.df_prices = pd.merge(self.df_prices,
                                    df_temp,
                                    how='left',
                                    left_index=True,
                                    right_index=True
                                    )
        
            

        #self.df_prices.to_excel('temp/dirty_prices.xlsx')


    def clean_price_matrix(self):

        self.__prepare_dirty_price_matrix__()

        # analyse holidays //TODO: visualize the holidays and export into excel
        holidays_days = self.df_prices.isna().sum(axis=1)
        holidays_days = holidays_days[holidays_days != 0]
        lst_holidays_tickers = list(self.df_prices.isna().sum(axis=0))

        # remove the holidays that are applicable to all the tickers (aka weekends)
        self.df_prices = self.df_prices[self.df_prices.count(axis=1) != 0]

        # fill the NaNs with the previously available price
        self.df_prices.fillna(method='ffill', inplace=True)
        self.df_prices.to_excel('portfolio/clean_prices.xlsx')

        return self.df_prices
    
    def return_matrix(self):

        df_log_prices_t_0 = self.df_prices.apply(lambda x: np.log(x)).shift(0)
        df_log_prices_t_1 = self.df_prices.apply(lambda x: np.log(x)).shift(1)

        self.df_log_returns = df_log_prices_t_0 - df_log_prices_t_1

        self.df_log_returns.dropna().to_excel('portfolio/logreturns matrix.xlsx')

    
    def correlation_matrix(self):

        df_corr = self.df_prices.corr()
        
        sns.heatmap(df_corr, xticklabels=df_corr.columns.values,yticklabels=df_corr.columns.values, annot=True)

        df_corr.to_excel('portfolio/correlation matrix.xlsx')

        plt.savefig('portfolio/correlation matrix.png')
        plt.clf()


    def mean_var_matrix(self):

        df_temp = pd.DataFrame(columns=['ticker','std','return'])

        df_temp['ticker'] = self.lst_tickers
        df_temp['bop_price'] = self.df_prices.iloc[0,:].values
        df_temp['eop_price'] = self.df_prices.iloc[-1,:].values
        df_temp['return'] = df_temp['eop_price'] / df_temp['bop_price'] - 1
        df_temp['std'] = self.df_log_returns.std().values
        
        df_temp.to_excel('portfolio/mean-variance.xlsx')


    def fit_gaussian_copula(self): # to enrich with Pearson, Kendall's Tau and Spearman Rho

        u1 = self.lst_cls_securities[0].df_stats['percentile']
        u2 = self.lst_cls_securities[1].df_stats['percentile']
        #u_combined = np.column_stack((u1,u2))
        df_combined = pd.merge(u1,u2,how='inner',left_index=True,right_index=True)

        copula = GaussianCopula()
        copula.fit_corr_param(df_combined)

        copula.plot_scatter()
        
        # plt.scatter(u1,u2)
        plt.savefig('portfolio/copula.png')
        plt.clf()


    def rebased_plot_30d(self):

        df_rebased = self.df_prices.iloc[0:30,'log_return']

# cls_asset1 = asset_performance(ticker='NVDA', b_from_FMP=True)
# cls_asset2 = asset_performance(ticker='LHA.DE', b_from_FMP=True)

# cls_portfolio = portfolio_performance()
# cls_portfolio.add_security(cls_asset1)
# cls_portfolio.add_security(cls_asset2)
# cls_portfolio.clean_price_matrix()
# cls_portfolio.correlation_matrix()
# cls_portfolio.fit_gaussian_copula()
# cls_portfolio.return_matrix()

