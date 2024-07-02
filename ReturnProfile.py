import pandas as pd
import numpy as np
import os
from datetime import timedelta, date
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from PIL import Image
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.distributions.empirical_distribution import ECDF

#TODO need to agree on the order of the time-series and/or implement the checks

class asset_performance:
    
    #TODO add optional tickers for references, like broad and narrow market + interest rate

    def __init__(self, ticker):
        
        #self.path = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\market data\\'
        self.path = 'C:\\Users\\ashve\\Desktop\\Projects\\market data\\'
        self.period = 252
        self.ticker = ticker
        
        lst_columns=['close','volume','log_return','vola30d','vola90d','vwap30d','vwap90d','adtv30d','adtv90d','rsi','obv','obv30d','obv90d','ma30d','ma90d','ewm']
        self.df_analysis = pd.DataFrame(columns=lst_columns)
        
        del lst_columns

        lst_columns=['date','close','log_return','rank_asc','rank_des','cdf']
        self.df_stats = pd.DataFrame(columns=lst_columns)

        del lst_columns

        self.download_data_for_a_ticker()
        self.compute_log_returns()
        
        self.compute_stats()
        
        #self.__fit_to_distributions__()
        self.compute_moving_average()
        self.compute_rsi()
        self.compute_vwap()
        self.compute_adtv()
        self.compute_obv()
        self.compute_vola()
        self.sort_prices()
        self.commpute_order_statistics()
        # self.add_distributions()
        self.save_xlsx()

    def download_data_for_a_ticker(self):
        
        df_temp = pd.read_csv(self.path + self.ticker + '.csv')
        df_temp = df_temp.set_index('date')
        
        # there is more data, adjClose is a starter
        self.df_analysis['close'] = df_temp[['adjClose']]
        self.df_analysis['volume'] = df_temp[['volume']]

        del df_temp
    
    def compute_log_returns(self):

        df_temp = pd.DataFrame(columns=['p(t-0)','p(t-1)','diff','return','log_return'])
        
        df_temp['p(t-0)'] = self.df_analysis['close'].shift(0)
        df_temp['p(t-1)'] = self.df_analysis['close'].shift(-1)
        df_temp['diff'] = df_temp['p(t-0)'] - df_temp['p(t-1)']
        df_temp['return'] = df_temp['diff'] / df_temp['p(t-1)']
        df_temp['log_return'] = df_temp['p(t-0)'].apply(np.log) - df_temp['p(t-1)'].apply(np.log)

        self.df_analysis['log_return'] = df_temp['log_return']

        #self.df_analysis.to_excel('xlsx/analysis.xlsx')

        del df_temp

    def compute_moving_average(self):
        
        df_temp = pd.DataFrame(columns=['close','ma30d','ma90d'])

        df_temp['close'] = self.df_analysis['close'].sort_index(ascending=True) # is sensitive to the sorting order
        
        df_temp['ma30d'] = df_temp['close'].rolling(window=30).mean()
        df_temp['ma90d'] = df_temp['close'].rolling(window=90).mean()
        df_temp['ewm'] = df_temp['close'].ewm(span=30,adjust=False).mean()

        self.df_analysis['ma30d'] = df_temp['ma30d']
        self.df_analysis['ma90d'] = df_temp['ma90d']
        self.df_analysis['ewm'] = df_temp['ewm']

        #self.df_analysis.to_excel('xlsx/analysis.xlsx')

        del df_temp

    def compute_rsi(self):

        df_temp = pd.DataFrame(columns=['close','diff','gain','loss','rsi'])
        
        df_temp['close'] = self.df_analysis['close'].sort_index(ascending=True) # is sensitive to the sorting order

        df_temp['diff'] = df_temp['close'].diff()

        df_temp['gain'] = (df_temp['diff'].where(df_temp['diff']>0,0)).rolling(window = 14).mean()
        df_temp['loss'] = (df_temp['diff'].where(df_temp['diff']<0,0)).rolling(window = 14).mean()

        df_temp['rsi'] = 100 - (100/(1+df_temp['gain']/df_temp['loss']))

        self.df_analysis['rsi'] = df_temp['rsi']

        #self.df_analysis.to_excel('xlsx/analysis.xlsx')

        del df_temp
        
    #    plt.subplot(121).plot(self.df_prices)
    #    plt.subplot(121).plot(rsi)

    #    plt.savefig("charts/RSI.png")


    #    loc_df_prices = self.df_prices.sort_index(ascending=True)
    #    loc_df_volumes = self.df_volumes.sort_index(ascending=True)

    #    plt.subplot(121).plot(loc_df_prices)
    #    plt.subplot(121).plot(loc_df_prices.rolling(window=30).mean())
    #    plt.subplot(121).plot(loc_df_prices.ewm(span=30, adjust=False).mean())
    #    plt.subplot(122).plot(loc_df_volumes)

    #    plt.savefig("charts/MA.png")

    def compute_vwap(self):
        
        df_temp = pd.DataFrame(columns=['close','volume','vwap30d','vwap90d'])

        df_temp['close'] = self.df_analysis['close']
        df_temp['volume'] = self.df_analysis['volume']

        n = self.dict_quick_stats['observation period (days)']
        lst_vwap_range = [30,90]
        
        def func_vwap(df):

            df_temp = df
            df_temp['weights'] = df_temp['volume'] / df_temp['volume'].sum()
            vwap = (df_temp['close'] * df_temp['weights']).sum()

            return vwap

        
        for i in range(0,n-lst_vwap_range[0]+1):

            df_temp['vwap30d'].iloc[i] = func_vwap(df_temp.iloc[i:i+lst_vwap_range[0]])
            #df_temp.loc[i,'vwap30d'] = func_vwap(df_temp.iloc[i:i+lst_vwap_range[0]])
            
        for i in range(0,n-lst_vwap_range[1]+1):

            df_temp['vwap90d'].iloc[i] = func_vwap(df_temp.iloc[i:i+lst_vwap_range[1]])
        
        
        self.df_analysis['vwap30d'] = df_temp['vwap30d']
        self.df_analysis['vwap90d'] = df_temp['vwap90d']

        #self.df_analysis.to_excel('xlsx/analysis.xlsx')

        del df_temp

    def compute_adtv(self):
        
        df_temp = pd.DataFrame(columns=['volume','adtv30d','adtv90d'])

        df_temp['volume'] = self.df_analysis['volume'].sort_index(ascending=True) # is sensitive to the sorting order

        df_temp['adtv30d'] = df_temp['volume'].rolling(window=30).mean()
        df_temp['adtv90d'] = df_temp['volume'].rolling(window=90).mean()

        self.df_analysis['adtv30d'] = df_temp['adtv30d']
        self.df_analysis['adtv90d'] = df_temp['adtv90d']
        
        #self.df_analysis.to_excel('xlsx/analysis.xlsx')

        del df_temp

    def compute_stats(self):
        
        df_temp = pd.DataFrame(columns=['log_return'])

        df_temp['log_return'] = self.df_analysis['log_return'].iloc[0:self.period-1]
        
        mean = df_temp.mean()
        std = df_temp.std()
        ann_period = 252
        obs_period = len(df_temp)
        vola = std * (obs_period/ann_period) ** 0.5
        high = df_temp.values.max()
        low = df_temp.values.min()
        skew = df_temp.skew()
        kurt = df_temp.kurt()
        
        self.dict_quick_stats = {"mean" : mean,"vola" : vola, "skew" : skew, "kurt" : kurt, "std" : std ,"observation period (days)" : obs_period}
        self.mean = mean
        self.std = std

        print(self.dict_quick_stats)

    def compute_obv(self): #does not work, rewrite as returns and volumes

        df_temp = pd.DataFrame(columns=['close','volume','log_return'])
        
        df_temp['close'] = self.df_analysis['close']
        df_temp['volume'] = self.df_analysis['volume']
        df_temp['log_return'] = self.df_analysis['log_return']

        df_temp['sign'] = (df_temp['log_return'] >= 0).replace({True:1, False:-1})
        
        df_temp['obv'] = df_temp['volume'] * df_temp['sign']
        
        df_temp.sort_index(ascending=True, inplace=True)

        df_temp['obv30d'] = df_temp['obv'].rolling(window=30).sum()
        df_temp['obv90d'] = df_temp['obv'].rolling(window=90).sum()

        self.df_analysis['obv'] = df_temp['obv']
        self.df_analysis['obv30d'] = df_temp['obv30d']
        self.df_analysis['obv90d'] = df_temp['obv90d']
        
        #self.df_analysis.to_excel('xlsx/analysis.xlsx')
        
        del df_temp

    def compute_vola(self):
        
        df_temp = pd.DataFrame(columns=['close','log_return','std30d','std90d','factor30d','factor90d','vola30d','vola90d'])

        df_temp['close'] = self.df_analysis['close']
        df_temp['log_return'] = self.df_analysis['log_return']

        df_temp.sort_index(ascending=True,inplace=True)

        df_temp['std30d'] = df_temp['log_return'].rolling(window=30).std()
        df_temp['std90d'] = df_temp['log_return'].rolling(window=90).std()

        df_temp['factor30d'] = (252/30) ** 0.5
        df_temp['factor90d'] = (252/90) ** 0.5

        df_temp['vola30d'] = df_temp['std30d'] * df_temp['factor30d']
        df_temp['vola90d'] = df_temp['std90d'] * df_temp['factor90d']

        self.df_analysis['vola30d'] = df_temp['vola30d']
        self.df_analysis['vola90d'] = df_temp['vola90d']

        self.df_analysis.to_excel('xlsx/analysis.xlsx')

        del df_temp

    def sort_prices(self):

        df_temp = pd.DataFrame(columns=['close','log_return','rank_asc','rank_des'])

        df_temp['log_return'] = self.df_analysis['log_return'].dropna()
        df_temp['close'] = self.df_analysis['close']

        df_temp.sort_values(by='log_return', inplace=True)
        
        n = len(df_temp)
        
        df_temp['rank_asc'] = range(1,n+1,1)
        df_temp['rank_des'] = range(n+1,1,-1)

        df_temp.reset_index(drop=False, inplace=True)
        df_temp.set_index(df_temp['rank_asc'], inplace=True)

        self.df_stats = df_temp

        #self.df_stats.to_excel('xlsx/stats.xlsx')

        del df_temp
        
    def commpute_order_statistics(self):
        
        df_temp = self.df_stats

        # add bins
        def bin_round(number): #atempt to generate comparable bins
            
            nearest_multiple = round(number / 0.0025) * 0.0025
            rounded_number = round(nearest_multiple, 3)
            
            return rounded_number

        df_temp['bin'] = df_temp['log_return'].map(lambda x: bin_round(x))
        self.df_stats['bin'] = df_temp['bin']

        # add percentile (which is )
        n = len(df_temp)
        df_temp['percentile'] = df_temp['rank_asc'].map(lambda x: x/(n+1))
        self.df_stats['percentile'] = df_temp['percentile']

        # add ecdf
        n = len(df_temp)
        df_temp['ecdf'] = [x * (1/(n)) for x in range(1,n+1,1)]
        self.df_stats['ecdf'] = df_temp['ecdf']
        
        # add z-score
        mean = self.dict_quick_stats['mean']
        scale = self.dict_quick_stats['std']

        df_temp['z_score'] = df_temp['log_return'].apply(lambda x: (x-mean)/scale)
        self.df_stats['z_score'] = df_temp['z_score']

        # add theoretical return
        df_temp['theo_return'] = stats.norm.cdf(df_temp['log_return'],mean,scale)
        self.df_stats['theo_return'] = df_temp['theo_return']

        # add theoretical percentile
        df_temp['theo_percentile'] = df_temp['z_score'].apply(lambda x: stats.norm.cdf(x))
        self.df_stats['theo_percentile'] = df_temp['theo_percentile']

        # add a column that highlights only z-score above absolute value of 1
        def filter_z_score(z_score, threshold):
            temp_result = 0
            if abs(z_score) >= threshold:
                temp_result = z_score
            
            return temp_result

        df_temp['significant_move'] = df_temp['z_score'].apply(lambda x: filter_z_score(x,1.5))
        self.df_stats['significant_move'] = df_temp['significant_move']

        # add bins
        min_bin = df_temp['bin'].min()
        max_bin = df_temp['bin'].max()
        tick = 0.0025
        n_bins = int((max_bin-min_bin)/tick)+1

        lst_bins = []

        for i in range(0,n_bins):

            lst_bins.append(round((min_bin + i * tick),5))

        df_pivot = pd.pivot_table(df_temp, values=['date'], columns=['bin'], aggfunc='count').T
        
        # save data with bins in a separate dataframe
        df_pdf = pd.DataFrame(index=lst_bins)
        df_pdf = pd.merge(df_pdf,df_pivot,left_index=True,right_index=True,how='left')
        df_pdf.fillna(0,inplace=True)

        self.df_pdf = df_pdf

        del df_temp, df_pdf, df_pivot

    # def add_distributions(self):

    #     mean = self.dict_quick_stats['mean']
    #     scale = self.dict_quick_stats['std']
    #     size = self.dict_quick_stats['observation period (days)']

    #     lst_index = list(range(1,size+1))
    #     df_temp = pd.DataFrame(columns=['norm_sample'], index=lst_index)

    #     rvs_pdf = stats.norm(loc=mean, scale=scale)
    #     lst_sample = sorted(list(rvs_pdf.rvs(size=size)))
        
    #     #rvs_cdf = stats.norm.cdf(lst_sample, mean, scale)

    #     df_temp['norm_sample'] = lst_sample
    #     #df_temp['norm_cdf'] = list(rvs_cdf)

    #     self.df_stats['norm_sample'] = df_temp['norm_sample']
    #     #self.df_stats['norm_cdf'] = df_temp['norm_cdf']

    #     del df_temp,mean, scale,size, lst_index,lst_sample, rvs_pdf #, rvs_cdf

    # def __fit_to_distributions__(self):
        
    #     rvs = stats.norm(loc=self.mean, scale=self.std)
    #     n = len(self.log_returns)
    #     self.theo_sample = rvs.rvs(size=n)
    #     self.real_sample = np.array(self.log_returns[self.ticker]) #without a ticker is better

    # def QQ_plot(self):
        
    #     stats.probplot(self.real_sample, dist="norm", plot=plt.subplot(121))
    #     stats.probplot(self.theo_sample, dist="norm", plot=plt.subplot(122))
        
    #     plt.savefig('charts/qq plot.png')
    #     plt.clf()
    
    # def __ecdf__(self, data):

    #     u_x = np.sort(data)
    #     n = len(u_x)
    #     u_y = np.arange(1, n + 1) / n
    #     return u_x, u_y
        
    ### TODO:looks wrong to me
    # def ECDF_plot(self):
        
    #     cdf_empirical = self.__ecdf__(self.real_sample)
        
    #     u_x = np.linspace(min(self.real_sample), max(self.real_sample), self.dict_quick_stats["observation period (days)"])
    #     u_y = stats.norm.cdf(u_x, self.mean, self.std)
    #     cdf_theoretical = [u_x,u_y]

    #     plt.plot(cdf_empirical[0],cdf_empirical[1])
    #     plt.plot(cdf_theoretical[0],cdf_theoretical[1])
        
    #     plt.savefig("charts/ecdf plot.png")

    def VaR_plot(self):
        
        self.L1W_returns = np.array(self.log_returns[self.ticker].iloc[0:4])
        self.L1M_returns = np.array(self.log_returns[self.ticker].iloc[0:21])

        #plot two histograms
        plt.subplot(121).hist(self.theo_sample, bins=50, alpha = 0.5, lw=0, fill=True)
        plt.subplot(122).hist(self.real_sample, bins=50, alpha = 0.5, lw=0, fill=True)

        plt.subplot(121).hist(self.L1W_returns, bins=50, alpha = 0.5, lw=0, color='red')
        plt.subplot(122).hist(self.L1M_returns, bins=50, alpha = 0.5, lw=0, color='green')

        #add bell curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin,xmax,100)
        p = stats.norm.pdf(x, self.mean,self.std)

        plt.subplot(121).plot(x, p, 'k', linewidth=0.5)
        plt.subplot(122).plot(x, p, 'k', linewidth=0.5)

        plt.savefig('charts/VaR plot.png')
        plt.clf()

    # at 99%
    def VaR_backtesting(self):
        tail_size = self.period * 0.01
        theo_VaR_sample = np.percentile(self.theo_sample, 0.01)
        theo_VaR_pdf = stats.norm.ppf(0.01, loc = self.mean, scale = self.std)
        real_VaR = np.percentile(self.real_sample, 0.01)

        li_behind_VaR_theo = self.real_sample[self.real_sample < theo_VaR_pdf]
        li_behind_VaR_real = self.real_sample[self.real_sample < real_VaR]

    def ACF_plot(self):
        
        plot_acf(self.log_returns).savefig('charts/ACF plot.png')

    def save_xlsx(self):

        self.df_analysis.to_excel('xlsx/analysis.xlsx')
        self.df_stats.to_excel('xlsx/stats.xlsx')
        self.df_pdf.to_excel('xlsx/pdf.xlsx')

df_check = asset_performance("ADBE")