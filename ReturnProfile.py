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
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from DownloadFMP import FMP_download


#TODO need to agree on the order of the time-series and/or implement the checks

class asset_performance:
    
    #TODO add optional tickers for references, like broad and narrow market + interest rate

    def __init__(self, ticker, b_from_FMP=False):
        
        #self.path = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\market data\\'
        self.path = 'C:\\Users\\ashve\\Desktop\\Projects\\market data\\'
        self.period = 252
        self.ticker = ticker
        self.b_from_FMP = b_from_FMP

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
        self.save_xlsx()
        self.prepare_a_chart()


    def download_data_for_a_ticker(self):
        
        if self.b_from_FMP:
        
            cls_FMP = FMP_download(self.ticker)
            df_temp = cls_FMP.df_output

        else:

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

        del df_temp


    def compute_rsi(self):

        df_temp = pd.DataFrame(columns=['close','diff','gain','loss','rsi'])
        
        df_temp['close'] = self.df_analysis['close'].sort_index(ascending=True) # is sensitive to the sorting order

        df_temp['diff'] = df_temp['close'].diff()

        df_temp['gain'] = (df_temp['diff'].where(df_temp['diff']>0,0)).rolling(window = 14).mean()
        df_temp['loss'] = (df_temp['diff'].where(df_temp['diff']<0,0)).rolling(window = 14).mean()

        df_temp['rsi'] = 100 - (100/(1+df_temp['gain']/df_temp['loss']))

        self.df_analysis['rsi'] = df_temp['rsi']

        del df_temp
        

    def compute_vwap(self):
        
        df_temp = pd.DataFrame(columns=['close','volume','vwap30d','vwap90d'])

        df_temp['close'] = self.df_analysis['close']
        df_temp['volume'] = self.df_analysis['volume']

        n = self.dict_quick_stats['observation period (days)']
        lst_vwap_range = [30,90]
        

        def func_vwap(df):

            df_temp = df.copy()
            #df_temp['weights'] = df_temp['volume'] / df_temp['volume'].sum()
            df_temp['weights'] = df_temp['volume'] / df_temp['volume'].sum()
            vwap = (df_temp['close'] * df_temp['weights']).sum()

            return vwap

        
        for i in range(0,n-lst_vwap_range[0]+1):

            df_temp.loc[i,'vwap30d'] = func_vwap(df_temp.iloc[i:i+lst_vwap_range[0]])
            #df_temp.loc[i,'vwap30d'] = func_vwap(df_temp.iloc[i:i+lst_vwap_range[0]])
            
        for i in range(0,n-lst_vwap_range[1]+1):

            df_temp.loc[i,'vwap90d'] = func_vwap(df_temp.iloc[i:i+lst_vwap_range[1]])
        
        
        self.df_analysis['vwap30d'] = df_temp['vwap30d']
        self.df_analysis['vwap90d'] = df_temp['vwap90d']

        del df_temp


    def compute_adtv(self):
        
        df_temp = pd.DataFrame(columns=['volume','adtv30d','adtv90d'])

        df_temp['volume'] = self.df_analysis['volume'].sort_index(ascending=True) # is sensitive to the sorting order

        df_temp['adtv30d'] = df_temp['volume'].rolling(window=30).mean()
        df_temp['adtv90d'] = df_temp['volume'].rolling(window=90).mean()

        self.df_analysis['adtv30d'] = df_temp['adtv30d']
        self.df_analysis['adtv90d'] = df_temp['adtv90d']

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

        df_temp['sign'] = np.where(df_temp['log_return'] >= 0,1,-1)

        df_temp['obv'] = df_temp['volume'] * df_temp['sign']
        
        df_temp.sort_index(ascending=True, inplace=True)

        df_temp['obv30d'] = df_temp['obv'].rolling(window=30).sum()
        df_temp['obv90d'] = df_temp['obv'].rolling(window=90).sum()

        self.df_analysis['obv'] = df_temp['obv']
        self.df_analysis['obv30d'] = df_temp['obv30d']
        self.df_analysis['obv90d'] = df_temp['obv90d']
        
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


    def prepare_a_chart(self):

        #general settings for the plot

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Consolas'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12  # Title font size
        plt.rcParams['axes.labelsize'] = 10  # Axis label font size
        plt.rcParams['xtick.labelsize'] = 10  # X-tick label font size
        plt.rcParams['ytick.labelsize'] = 10  # Y-tick label font size
        plt.rcParams['figure.figsize'] = [19, 12]

        plt.gcf().autofmt_xdate()

        fig, ax = plt.subplots(nrows=3, ncols=3)

        fig.suptitle('Share price performance: ' + self.ticker, fontsize=16)
        #plt.figure(figsize=(30,30))

        # plot price and moving average
        df_plot1 = self.df_analysis[['close','ma30d']].sort_index(ascending=True)
        u_x = df_plot1.index
        u_y1 = df_plot1.iloc[:,0]
        u_y2 = df_plot1.iloc[:,1]

        ax[0,0].plot(u_y1,'r')
        ax[0,0].plot(u_y2,'b')
        ax[0,0].set_title('Price performance')
        ax[0,0].xaxis.set_major_locator(mdates.MonthLocator())
        ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


        # plot volume and vola

        df_plot2 = self.df_analysis[['volume','vola30d']].sort_index(ascending=True)
        u_x = df_plot2.index
        u_y1 = df_plot2.iloc[:,0]
        u_y2 = df_plot2.iloc[:,1]

        ax[1,0].bar(u_x,u_y1)
        ax[1,0].twinx().plot(u_y2,'b')
        ax[1,0].set_title('Volume and vola')
        ax[1,0].xaxis.set_major_locator(mdates.MonthLocator())
        ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


        # plot significant moves

        df_plot3 = self.df_stats[['date','significant_move']].sort_index(ascending=True)
        df_plot3.reset_index(inplace=True)
        df_plot3.set_index(df_plot3['date'],inplace=True)
        df_plot3.sort_index(ascending=True, inplace=True)

        u_x = df_plot3.index
        u_y1 = df_plot3.iloc[:,2]

        ax[2,0].bar(u_x,u_y1)
        ax[2,0].set_title('Significant moves')
        ax[2,0].xaxis.set_major_locator(mdates.MonthLocator())
        ax[2,0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        # plot histogram and VaR

        df_plot4 = self.df_stats[['log_return','date']]

        # plot log returns
        u_x = df_plot4.index
        u_y1 = df_plot4['log_return']

        ax[0,1].hist(u_y1,bins = 50)

        # plot recent log returns (1 Month)
        df_temp = df_plot4.sort_values(by='date',ascending=False)
        arr_L1M_returns = np.array(df_temp['log_return'].iloc[0:21])

        ax[0,1].hist(arr_L1M_returns, bins=50, color = "red")

        # plot the normal distribution on top
        x_min = df_plot4['log_return'].min()
        x_max = df_plot4['log_return'].max()
        n = len(df_plot4)
        mean = self.dict_quick_stats['mean']
        scale = self.dict_quick_stats['std']

        x = np.linspace(x_min,x_max,n)
        p = stats.norm.pdf(x, mean,scale)
        ax[0,1].plot(x, p, 'k', linewidth=0.5)

        # add empirical VaR
        emp_VaR = np.percentile(df_plot4['log_return'], 1)
        ax[0,1].axvline(emp_VaR, linestyle='dashed')

        # add theoretical VaR
        theo_VaR = stats.norm.ppf(0.01, loc=mean, scale=scale)
        ax[0,1].axvline(theo_VaR, linestyle='dashed', color="black")

        ax[0,1].set_title('Histogram and VaR')


        # chart a Q-Q plot

        df_plot5 = self.df_stats[['percentile','theo_percentile']]

        u_x = df_plot5['theo_percentile']
        u_y1 = df_plot5['theo_percentile']
        u_y2 = df_plot5['percentile']

        ax[1,1].plot(u_x,u_y1)
        ax[1,1].scatter(u_x,u_y2,s=2, color = "black")
        ax[1,1].set_title('Q-Q plot')


        # chart a violin plot

        df_plot6 = self.df_stats['log_return']

        ax[2,1].violinplot(df_plot6)
        ax[2,1].set_title('Violin plot')


        # chart an ACF

        df_plot7 = self.df_analysis['log_return'].dropna()

        plot_acf(df_plot7,ax = ax[0,2])


        # chart cdf

        df_plot8 = self.df_stats[['log_return','ecdf']]

        # theoretical
        x = np.linspace(x_min,x_max,n) #see chart above
        p = stats.norm.cdf(x, mean,scale)
        ax[1,2].plot(x, p, 'k', linewidth=0.5)

        # empirical
        u_x = df_plot8.iloc[:,0]
        u_y = df_plot8.iloc[:,1]
        ax[1,2].plot(u_x, u_y, 'k', linewidth=0.5)

        ax[1,2].set_title('ECDF')


        # chart some technical indicator (rsi as a placeholder)

        df_plot9 = self.df_analysis[['log_return','rsi']].sort_index(ascending=True)

        u_x = df_plot9.index
        u_y1 = df_plot9.iloc[:,0]
        u_y2 = df_plot9.iloc[:,1]

        ax[2,2].bar(u_x,u_y1)
        ax[2,2].twinx().plot(u_y2,'b')
        ax[2,2].set_title('Returns and RSI')
        ax[2,2].xaxis.set_major_locator(mdates.MonthLocator())
        ax[2,2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        plt.tight_layout()
        plt.savefig('charts/' + self.ticker + '.png')

    # at 99%
    # def VaR_backtesting(self):
    #     tail_size = self.period * 0.01
    #     theo_VaR_sample = np.percentile(self.theo_sample, 0.01)
    #     theo_VaR_pdf = stats.norm.ppf(0.01, loc = self.mean, scale = self.std)
    #     real_VaR = np.percentile(self.real_sample, 0.01)

    #     li_behind_VaR_theo = self.real_sample[self.real_sample < theo_VaR_pdf]
    #     li_behind_VaR_real = self.real_sample[self.real_sample < real_VaR]

    # def ACF_plot(self):
        
    #     plot_acf(self.log_returns).savefig('charts/ACF plot.png')

    def save_xlsx(self):

        self.df_analysis.to_excel(f'xlsx/{self.ticker}_analysis.xlsx')
        self.df_stats.to_excel(f'xlsx/{self.ticker}_stats.xlsx')
        #self.df_pdf.to_excel('xlsx/pdf.xlsx')

#df_check = asset_performance("NVDA")