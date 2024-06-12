import pandas as pd
import numpy as np
import os
from datetime import timedelta, date
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from PIL import Image
from statsmodels.graphics.tsaplots import plot_acf

class asset_performance:
    
    #TODO add optional tickers for references, like broad and narrow market + interest rate

    def __init__(self, ticker):
        self.path = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\market data\\'
        self.period = 252
        self.ticker = ticker
        self.download_data_for_a_ticker()
        self.compute_log_returns()
        self.compute_stats()
        self.generate_reference_distribution()
        self.__fit_to_distributions__()

    def download_data_for_a_ticker(self):
        df_temp = pd.read_csv(self.path + self.ticker + '.csv')
        df_temp = df_temp.set_index('date')
        # there is more data, adjClose is a starter
        df_temp = df_temp[['adjClose']]
        df_temp.rename(columns={'adjClose': self.ticker}, inplace=True)
        self.market_data = df_temp
    
    def compute_log_returns(self):
        df_temp = -(self.market_data.shift(0).apply(np.log) - self.market_data.shift(1).apply(np.log)) #idk why I have to put a minus in front
        self.log_returns = df_temp.shift(-1).dropna()

    def compute_stats(self):
        df_temp = self.log_returns.iloc[0:self.period-1]
        self.mean_log_return = df_temp.mean()
        self.vola = df_temp.std()
        self.high = df_temp.values.max()
        self.low = df_temp.values.min()

        print(f"Mean return is {self.mean_log_return}. Vola is {self.vola}. Highest return is {self.high}. Lowest - {self.low}.")

    def generate_reference_distribution(self):
        np.random.normal(self.mean_log_return, self.vola, self.period)

    def __fit_to_distributions__(self):
        rvs = stats.norm(loc=self.mean_log_return, scale=self.vola)
        
        self.theo_sample = rvs.rvs(size=self.period)
        self.real_sample = np.array(self.log_returns[self.ticker])

    def QQ_plot(self):
        
        stats.probplot(self.real_sample, dist="norm", plot=plt.subplot(121))
        stats.probplot(self.theo_sample, dist="norm", plot=plt.subplot(122))
        
        plt.savefig('qq plot.png')
        plt.clf()

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
        p = stats.norm.pdf(x, self.mean_log_return,self.vola)

        plt.subplot(121).plot(x, p, 'k', linewidth=0.5)
        plt.subplot(122).plot(x, p, 'k', linewidth=0.5)

        plt.savefig('VaR plot.png')
        plt.clf()

    # at 99%
    def VaR_backtesting(self):
        tail_size = self.period * 0.01
        theo_VaR_sample = np.percentile(self.theo_sample, 0.01)
        theo_VaR_pdf = stats.norm.ppf(0.01, loc = self.mean_log_return, scale = self.vola)
        real_VaR = np.percentile(self.real_sample, 0.01)

        li_behind_VaR_theo = self.real_sample[self.real_sample < theo_VaR_pdf]
        li_behind_VaR_real = self.real_sample[self.real_sample < real_VaR]

        print(tail_size)
        print(li_behind_VaR_theo)
        print(li_behind_VaR_real)
        print(theo_VaR_pdf)
        print(theo_VaR_sample)
        print(real_VaR)

    def ACF_plot(self):
        
        plot_acf(self.log_returns).savefig('ACF plot.png')

df_check = asset_performance("ADSK")

df_check.ACF_plot()
