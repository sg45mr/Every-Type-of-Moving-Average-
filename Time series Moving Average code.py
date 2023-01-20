# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:35:19 2023

@author: swaru
"""

import seaborn as sns
import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime 
import pandas as pd
mpl.rcParams['figure.figsize']=(10,8)
mpl.rcParams["axes.grid"]=False

def parse(x):
    return datetime.strptime(x,"%m/%d/%Y")

df=pd.read_csv('https://raw.githubusercontent.com/srivatsan88/YoutubeLI/master/dataset/electricity_consumption.csv',parse_dates=['Bill_Date'],date_parser=parse)
df.head()

print("Rows        : ",df.shape[0])
print("Columns     : ",df.shape[1])
print("\nFeatures  : ",df.columns.tolist())
print("\nMissing values : ",df.isnull().any())
print("\n Unique value  : ",df.nunique())

bill_df=df.set_index("Bill_Date")
bill_df.head()

bill_2018=bill_df['2016':'2018'][['Billed_amount']]
bill_2018

bill_2018['Billed_amount'].rolling(window=3).mean() 
bill_2018["MA_rolling_3"]=bill_2018['Billed_amount'].rolling(window=3).mean().shift(1)
bill_2018
# The reason i'm using shift(1) function is basically i'm taking current value and previous two value and i'm predicting 
# next value

bill_2018.plot()

def w_MA(weights):
    def calc(x):
        return (weights*x).mean()
    return calc

bill_2018["Billed_amount"].rolling(window=3).apply(w_MA(np.array([0.5,1,1.5])))

bill_2018["W_MA_rolling_3"]=bill_2018['Billed_amount'].rolling(window=3).apply(w_MA(np.array([0.5,1,1.5]))).shift(1)
bill_2018

bill_2018.plot()

bill_2018['Billed_amount'].ewm(span=3,adjust=False,min_periods=0).mean()

bill_2018["EMA_window_3"]=bill_2018['Billed_amount'].ewm(span=3,adjust=False,min_periods=0).mean().shift(1)
bill_2018

bill_2018.plot()

bill_2018["Billed_amount"].ewm(alpha=0.7,adjust=False,min_periods=3).mean()

bill_2018["ESMA_window_3_7"]=bill_2018['Billed_amount'].ewm(alpha=0.7,adjust=False,min_periods=3).mean().shift(1)
bill_2018

bill_2018.plot()


bill_2018["ESMA_window_3_3"]=bill_2018['Billed_amount'].ewm(alpha=0.3,adjust=False,min_periods=3).mean().shift(1)
bill_2018

bill_2018[["Billed_amount","ESMA_window_3_3","ESMA_window_3_7"]].plot()


RMSE_MA_rolling_3=((bill_2018['Billed_amount']-bill_2018["MA_rolling_3"])**2).mean()**0.5
RMSE_W_MA_rolling_3=((bill_2018['Billed_amount']-bill_2018["W_MA_rolling_3"])**2).mean()**0.5
RMSE_EMA_window_3=((bill_2018['Billed_amount']-bill_2018["EMA_window_3"])**2).mean()**0.5
RMSE_ESMA_window_3_7=((bill_2018['Billed_amount']-bill_2018["ESMA_window_3_7"])**2).mean()**0.5




























