import pandas as pd
from pandas import DataFrame
import os
from DownloadFMP import FMP_download
import matplotlib.pyplot as plt
import seaborn as sns


class financial_performance:

    def __init__(self, ticker:str, b_from_FMP:bool=True):

        self.path:str = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\'
        #self.path:str = 'C:\\Users\\ashve\\Desktop\\Projects\\market data\\'
        self.ticker:str = ticker
        self.b_from_FMP:bool = b_from_FMP
        self.str_folder:str = f'profiles/{ticker}/'
        os.makedirs(self.str_folder, exist_ok=True)
        self.FYears:list = ['2015','2016','2017','2018','2019','2020','2021','2022','2023']
        self.scale = 1000000 # convert numbers into millions

        self.df_inputs:DataFrame = pd.DataFrame(columns=['revenue',
                                                         'cogs',
                                                         'sg&a',
                                                         'd&a',
                                                         'r&d',
                                                         'capex',
                                                         'taxes',
                                                         'interest income',
                                                         'interest expense',
                                                         'current assets',
                                                         'current liabilities',
                                                         'change in wc',
                                                         'buybacks',
                                                         'dividends',
                                                         'receivables',
                                                         'payables',
                                                         'inventory',
                                                         'cash',
                                                         'st investments',
                                                         'st debt',
                                                         'lt debt',
                                                         'leasing'],
                                                index=self.FYears)
        self.df_inputs.index.name = 'CY'


    def download_statements(self):
        
        str_path:str = self.str_folder

        self.df_BS:DataFrame = FMP_download(self.ticker).financial_statement('BS')
        self.df_BS.to_excel(str_path + '_BS.xlsx')

        self.df_IS:DataFrame = FMP_download(self.ticker).financial_statement('IS')
        self.df_IS.to_excel(str_path + '_IS.xlsx')

        self.df_CF:DataFrame = FMP_download(self.ticker).financial_statement('CF')
        self.df_CF.to_excel(str_path + '_CF.xlsx')

        self.df_BS:DataFrame = FMP_download(self.ticker).financial_statement('DIV')
        self.df_BS.to_excel(str_path + '_DIV.xlsx')
        

    def calcs(self):

        df_BS:DataFrame = self.df_BS[['calendarYear',
                                      'totalCurrentAssets',
                                      'totalCurrentLiabilities',
                                      'netReceivables',
                                      'accountPayables',
                                      'inventory',
                                      'cashAndCashEquivalents',
                                      'shortTermInvestments',
                                      'shortTermDebt',
                                      'longTermDebt',
                                      'capitalLeaseObligations']]

        df_BS.set_index('calendarYear',inplace=True)
        df_BS.index.name = 'CY'

        self.df_inputs[['current assets','current liabilities','receivables','payables','inventory','cash','st investments','st debt','lt debt','leasing']] = df_BS / self.scale

        df_IS:DataFrame = self.df_IS[['calendarYear',
                                      'revenue',
                                      'costOfRevenue',
                                      'sellingGeneralAndAdministrativeExpenses',
                                      'depreciationAndAmortization',
                                      'researchAndDevelopmentExpenses',
                                      'incomeTaxExpense',
                                      'interestIncome',
                                      'interestExpense']]

        df_IS.set_index('calendarYear',inplace=True)
        df_IS.index.name = 'CY'

        self.df_inputs[['revenue','cogs','sg&a','d&a','r&d','taxes','interest income','interest expense']] = df_IS / self.scale

        df_CF:DataFrame = self.df_CF[['calendarYear',
                                      'investmentsInPropertyPlantAndEquipment',
                                      'changeInWorkingCapital',
                                      'commonStockRepurchased',
                                      'dividendsPaid']]

        df_CF.set_index('calendarYear',inplace=True)
        df_CF.index.name = 'CY'

        self.df_inputs[['capex','change in wc','buybacks','dividends']] = df_CF / self.scale

        df_inputs:DataFrame = self.df_inputs
        
        df_calcs:DataFrame = pd.DataFrame(columns=['EBIT',
                                                   'op income',
                                                   'working capital',
                                                   'capital distributions',
                                                   'interest expense',
                                                   'sales growth',
                                                   'gp margin',
                                                   'EBITDA margin',
                                                   'EBIT margin',
                                                   'wc turnover',
                                                   'inventory turnover',
                                                   'gross profit',
                                                   'EBITDA',
                                                   'net debt',
                                                   'leverage',
                                                   'wc ratio',
                                                   'solvency ratio'],
                                          index=self.FYears)
        
        df_calcs.index.name = 'CY'

        df_calcs['EBIT'] = df_inputs['revenue'] - df_inputs['cogs'] - df_inputs['sg&a']
        df_calcs['EBIT margin'] = df_calcs['EBIT'] / df_inputs['revenue']
        df_calcs['interest expense'] = df_inputs['interest expense'] - df_inputs['interest income']
        df_calcs['op income'] = df_calcs['EBIT'] - df_inputs['taxes']
        df_calcs['working capital'] = df_inputs['current assets'] - df_inputs['current liabilities']
        df_calcs['capital distributions'] = df_inputs['dividends'] + df_inputs['buybacks'].abs()
        df_calcs['EBITDA'] = df_calcs['EBIT'] + df_inputs['d&a']
        df_calcs['gross profit'] = df_inputs['revenue'] - df_inputs['cogs']
        df_calcs['gp margin'] = df_calcs['gross profit'] / df_inputs['revenue']
        df_calcs['EBITDA margin'] = df_calcs['EBITDA'] / df_inputs['revenue']
        df_calcs['sales growth'] = df_inputs['revenue'] / df_inputs['revenue'].shift(1)
        df_calcs['net debt'] = df_inputs['cash'] - df_inputs['st debt'] - df_inputs['lt debt'] - df_inputs['leasing']
        df_calcs['leverage'] = df_calcs['net debt'] / df_calcs['EBITDA']
        df_calcs['interest cover'] = df_calcs['EBIT'] / df_calcs['interest expense']
        df_calcs['wc ratio'] = df_inputs['current assets'] / df_inputs['current liabilities']
        df_calcs['solvency ratio'] = (df_calcs['op income'] + df_inputs['d&a']) / df_calcs['net debt']
        df_calcs['wc turnover'] = df_inputs['revenue'] / (0.5*(df_calcs['working capital'] + df_calcs['working capital'].shift(1)))
        df_calcs['inventory turnover'] = df_inputs['revenue'] / (0.5*(df_inputs['inventory'] + df_inputs['inventory'].shift(1)))

        self.df_calcs:DataFrame = df_calcs
        df_inputs.to_excel('profiles/ADBE/inputs.xlsx')
        df_calcs.to_excel('profiles/ADBE/calcs.xlsx')

    def slide_helicopter_view(self):

        df_inputs:DataFrame = self.df_inputs
        df_calcs:DataFrame = self.df_calcs
        df_text:DataFrame = pd.DataFrame()

        
cls_ADBE = financial_performance('ADBE')
cls_ADBE.download_statements()
cls_ADBE.calcs()

    

