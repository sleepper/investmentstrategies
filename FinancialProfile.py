import pandas as pd
from pandas import DataFrame
import os
from DownloadFMP import FMP_download

class financial_performance:

    def __init__(self, ticker:str, b_from_FMP:bool=True):

        self.path:str = 'c:\\Users\\top kek\\Desktop\\Python\\2_External APIs\\'
        #self.path:str = 'C:\\Users\\ashve\\Desktop\\Projects\\market data\\'
        self.ticker:str = ticker
        self.b_from_FMP:bool = b_from_FMP
        self.str_folder:str = f'profiles/{ticker}/'
        os.makedirs(self.str_folder, exist_ok=True)
        self.FYears:list = ['2017','2018','2019','2020','2021','2022','2023']

    def download_statements(self):
        
        str_path:str = self.str_folder

        self.df_BS:DataFrame = FMP_download(self.ticker).financial_statement('BS')
        #self.df_BS.to_excel(str_path + '_BS.xlsx')

        self.df_IS:DataFrame = FMP_download(self.ticker).financial_statement('IS')
        #self.df_IS.to_excel(str_path + '_IS.xlsx')

        self.df_CF:DataFrame = FMP_download(self.ticker).financial_statement('CF')
        #self.df_CF.to_excel(str_path + '_CF.xlsx')
        

    def cash_flow_dynamics(self):

        df_inputs:DataFrame = pd.DataFrame(columns=['revenue',
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
                                                    'dividends'],
                                            index=self.FYears)
        df_inputs.index.name = 'CY'

        df_BS:DataFrame = self.df_BS[['calendarYear',
                                      'totalCurrentAssets',
                                      'totalCurrentLiabilities']]

        df_BS.set_index('calendarYear',inplace=True)
        df_BS.index.name = 'CY'

        df_inputs[['current assets','current liabilities']] = df_BS

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

        df_inputs[['revenue','cogs','sg&a','d&a','r&d','taxes','interest income','interest expense']] = df_IS

        df_CF:DataFrame = self.df_CF[['calendarYear',
                                      'investmentsInPropertyPlantAndEquipment',
                                      'changeInWorkingCapital',
                                      'commonStockRepurchased',
                                      'dividendsPaid']]

        df_CF.set_index('calendarYear',inplace=True)
        df_CF.index.name = 'CY'

        df_inputs[['capex','change in wc','buybacks','dividends']] = df_CF        

        df_calcs:DataFrame = pd.DataFrame(columns=['EBIT',
                                                   'working capital',
                                                   'capital distributions',
                                                   'interest expense'],
                                          index=self.FYears)
        
        df_calcs.index.name = 'CY'

        df_calcs['EBIT'] = df_inputs['revenue'] - df_inputs['cogs'] - df_inputs['sg&a']
        df_calcs['interest expense'] = df_inputs['interest expense'] - df_inputs['interest income']
        df_calcs['working capital'] = df_inputs['current assets'] - df_inputs['current liabilities']
        df_calcs['capital distributions'] = df_inputs['dividends'] + df_inputs['buybacks'].abs()

        df_inputs.to_excel('inputs.xlsx')
        df_calcs.to_excel('calcs.xlsx')


cls_ADBE = financial_performance('ADBE')
cls_ADBE.download_statements()
cls_ADBE.cash_flow_dynamics()

    

