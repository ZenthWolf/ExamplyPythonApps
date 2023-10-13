#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:26:43 2023

@author: zenth
"""

#%% Install Alpha Vantage API
%pip install alpha_vantage

#%% Imports
from alpha_vantage.timeseries import TimeSeries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%% Read API Key
with open('api_key.txt') as f:
    API_Key = f.read()

API_Key = API_Key.strip()
API_Key

#%% Using Python Package

#%% TimeSeries

ts1 = TimeSeries(key=API_Key)

#%% Get Unity Stock monthly
ts1.get_monthly("U")

#%% Get Unity stock weekly

ts1.get_weekly("U")

#%% Intraday stock

ts1.get_intraday("U")

#%% Using Requests

#%% Get monthly time series

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=u&apikey={API_Key}'

r = requests.get(url)
data = r.json()

print(data)

#%% Get Weekly time series

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=u&apikey={API_Key}'

r = requests.get(url)
data = BeautifulSoup(r.content)

print(data)

#%% Get Weekly time series

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=u&interval=60min&apikey={API_Key}'

r = requests.get(url)
data = BeautifulSoup(r.content)

print(data)

#%% Utilizing pandas with requests

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=u&apikey={API_Key}&datatype=csv'

r = requests.get(url).content
data = pd.read_csv(io.StringIO(r.decode('utf-8')))

print(data)

#%% Utilizing pandas with python package

unity1, meta_data = ts1.get_intraday("U")

print(meta_data)

print(unity1)

#%% Transform to dataframe

df_unity1 = pd.DataFrame(unity1).transpose().reset_index()

df_unity1.head()

#%% Dataframe format directly

ts2 = TimeSeries(key=API_Key, output_format='pandas')

df_Unity2, metadata = ts2.get_intraday("U", outputsize="full")

print(df_Unity2.reset_index())

print(meta_data)

#%% Alpha Vantage Functionality

#%% Income Statement
url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=U&apikey={API_Key}'
r = requests.get(url)
fd = BeautifulSoup(r.content)

print(fd)

#%% Cash flow statement
url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol=U&apikey={API_Key}'
r = requests.get(url)
fd = BeautifulSoup(r.content)

print(fd)

#%% Foreign Currency Exchange Rate

#Exchange rate from USD to JPY
url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=JPY&apikey={API_Key}'
r = requests.get(url)
fx = BeautifulSoup(r.content)

print(fx)

#%% Daily Foreign Exchange Rate

url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=JPY&apikey={API_Key}'
r = requests.get(url)
fx = BeautifulSoup(r.content)

print(fx)

#%% Cryptocurrency weekly exchange rate

url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_WEEKLY&symbol=BTC&market=USD&apikey={API_Key}'
r = requests.get(url)
fx = BeautifulSoup(r.content)

print(fx)

#%% Quarterly US GDP

url = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={API_Key}'
r = requests.get(url)
ei = BeautifulSoup(r.content)

print(ei)

#%% Monthly US Consumer Price Index

# 100 between 1982-1984

url = f'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={API_Key}'
r = requests.get(url)
ei = BeautifulSoup(r.content)

print(ei)

#%% Monthly US Unemployment

url = f'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={API_Key}&datatype=csv'
#r = requests.get(url)
#ei = BeautifulSoup(r.content)
r = requests.get(url).content
data = pd.read_csv(io.StringIO(r.decode('utf-8')))

print(data.head)

#%% News and Sentiments

url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=U&apikey={API_Key}'
r = requests.get(url)
news = BeautifulSoup(r.content)

print(news)

#%% Simple Moving Average (Weekly, averaged over last 10)

url = f'https://www.alphavantage.co/query?function=SMA&symbol=U&interval=weekly&time_period=10&series_type=open&apikey={API_Key}'
r = requests.get(url)
ti = BeautifulSoup(r.content)

print(ti)

#%% Weighted Moving Average (Weekly, averaged over last 10)

url = f'https://www.alphavantage.co/query?function=WMA&symbol=U&interval=weekly&time_period=10&series_type=open&apikey={API_Key}'
r = requests.get(url)
ti = BeautifulSoup(r.content)

print(ti)

#%% Rate of Change Ratios

# Percentage change currently vs 10 periods prior
url = f'https://www.alphavantage.co/query?function=ROCR&symbol=U&interval=daily&time_period=10&series_type=close&apikey={API_Key}'
r = requests.get(url)
ti = BeautifulSoup(r.content)

print(ti)

#%% Bollinger Bands


#SMA along with range of values (3 stdev here)
url = f'https://www.alphavantage.co/query?function=BBANDS&symbol=U&interval=weekly&time_period=5&series_type=close&nbdevup=3&nbdevdn=3&apikey={API_Key}'
r = requests.get(url)
ti = BeautifulSoup(r.content)

print(ti)

#%% Challenge: Analyze HASbro stock

#%% Get Monthly data

ts1.get_monthly('HAS')

#%% Get Weekly data

ts1.get_weekly('HAS')

#%% Pull income statement

url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=HAS&apikey={API_Key}'
r = requests.get(url)
fd = BeautifulSoup(r.content)

print(fd)

#%% Bollinger Bands for Daily price over 10 periods- open stock value

url = f'https://www.alphavantage.co/query?function=BBANDS&symbol=HAS&interval=daily&time_period=10&series_type=open&nbdevup=3&nbdevdn=3&apikey={API_Key}'
r = requests.get(url)
ti = BeautifulSoup(r.content)

print(ti)