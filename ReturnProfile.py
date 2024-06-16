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
        self.path = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\market data\\'
        self.period = 252
        self.ticker = ticker
        self.download_data_for_a_ticker()
        self.compute_log_returns()
        self.compute_stats()
        self.__fit_to_distributions__()

    def download_data_for_a_ticker(self):
        
        df_temp = pd.read_csv(self.path + self.ticker + '.csv')
        df_temp = df_temp.set_index('date')
        
        # there is more data, adjClose is a starter
        df_prices = df_temp[['adjClose']]
        df_prices.rename(columns={'adjClose': self.ticker}, inplace=True)
        self.df_prices = df_prices

        df_volumes = df_temp[['volume']]
        self.df_volumes = df_volumes

    
    def compute_log_returns(self):
        df_temp = -(self.df_prices.shift(0).apply(np.log) - self.df_prices.shift(1).apply(np.log)) #idk why I have to put a minus in front
        self.log_returns = df_temp.shift(-1).dropna()

    def compute_stats(self):
        df_temp = self.log_returns.iloc[0:self.period-1]
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

    def __fit_to_distributions__(self):
        
        rvs = stats.norm(loc=self.mean, scale=self.std)
        n = len(self.log_returns)
        self.theo_sample = rvs.rvs(size=n)
        self.real_sample = np.array(self.log_returns[self.ticker]) #without a ticker is better

    def QQ_plot(self):
        
        stats.probplot(self.real_sample, dist="norm", plot=plt.subplot(121))
        stats.probplot(self.theo_sample, dist="norm", plot=plt.subplot(122))
        
        plt.savefig('qq plot.png')
        plt.clf()
    
    def __ecdf__(self, data):

        u_x = np.sort(data)
        n = len(u_x)
        u_y = np.arange(1, n + 1) / n
        return u_x, u_y
        
    ### TODO:looks wrong to me
    def ECDF_plot(self):
        
        cdf_empirical = self.__ecdf__(self.real_sample)
        
        u_x = np.linspace(min(self.real_sample), max(self.real_sample), self.dict_quick_stats["observation period (days)"])
        u_y = stats.norm.cdf(u_x, self.mean, self.std)
        cdf_theoretical = [u_x,u_y]

        plt.plot(cdf_empirical[0],cdf_empirical[1])
        plt.plot(cdf_theoretical[0],cdf_theoretical[1])
        
        plt.savefig("ecdf plot.png")

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

        plt.savefig('VaR plot.png')
        plt.clf()

    # at 99%
    def VaR_backtesting(self):
        tail_size = self.period * 0.01
        theo_VaR_sample = np.percentile(self.theo_sample, 0.01)
        theo_VaR_pdf = stats.norm.ppf(0.01, loc = self.mean, scale = self.std)
        real_VaR = np.percentile(self.real_sample, 0.01)

        li_behind_VaR_theo = self.real_sample[self.real_sample < theo_VaR_pdf]
        li_behind_VaR_real = self.real_sample[self.real_sample < real_VaR]

        # print(tail_size)
        # print(li_behind_VaR_theo)
        # print(li_behind_VaR_real)
        # print(theo_VaR_pdf)
        # print(theo_VaR_sample)
        # print(real_VaR)

    def ACF_plot(self):
        
        plot_acf(self.log_returns).savefig('ACF plot.png')

    def moving_average(self):
        
        loc_df_prices = self.df_prices.sort_index(ascending=True)
        loc_df_volumes = self.df_volumes.sort_index(ascending=True)

        plt.subplot(121).plot(loc_df_prices)
        plt.subplot(121).plot(loc_df_prices.rolling(window=30).mean())
        plt.subplot(121).plot(loc_df_prices.ewm(span=30, adjust=False).mean())
        plt.subplot(122).plot(loc_df_volumes)

        plt.savefig("MA.png")

    def rsi(self):

        df_diffs = self.df_prices.sort_index(ascending=True).diff()

        df_gains = (df_diffs.where(df_diffs>0,0)).rolling(window = 14).mean()
        df_loss = (df_diffs.where(df_diffs<0,0).abs()).rolling(window = 14).mean()

        rsi = 100 - (100/(1+df_gains/df_loss))
        
        plt.subplot(121).plot(self.df_prices)
        plt.subplot(121).plot(rsi)

        plt.savefig("RSI.png")

    def vwap(self):
        
        self.df_vwap = pd.merge(self.df_prices,self.df_volumes,left_index=True, right_index=True, how='left')
        self.df_vwap.columns = ['p','q']
        n = self.dict_quick_stats['observation period (days)']
        vwap_range = 30
        self.df_vwap['vwap'] = None
        
        for i in range(0,n-vwap_range+1):
            
            df_temp = self.df_vwap.iloc[i:i+vwap_range]
            df_temp['weights'] = df_temp.q / df_temp.q.sum()
            vwap = (df_temp.p * df_temp.weights).sum()
            self.df_vwap['vwap'].iloc[i] = vwap
        
        # self.df_vwap.to_excel('vwap.xlsx')

df_check = asset_performance("ADSK")
print(df_check.df_vwap)