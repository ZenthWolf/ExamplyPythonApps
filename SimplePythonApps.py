#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:55:12 2023

@author: zenth
"""

#%% Cell Imports

import requests
from bs4 import BeautifulSoup

#%% Scrape WisdomPetMedicine (fictional business)

response = requests.get('https://wisdompetmed.com')

response.url           # Check URL is correct
response.status_code   # Check website Status is good 2xx
response.headers       # Check headers
response.content       # HTML in byte data
response.text          # HTML as UTF-8 string (More readable, might less accurate)

#%% Parsing

soup = BeautifulSoup(response.text, features="lxml")
print('\nHTML')
print(soup.prettify() )
business_name = soup.find('title')
print('\nBusiness Name:')
print(business_name)
services_list = soup.find_all('article')
print('\nList of Services')
print(services_list)
business_phone = soup.find('span', class_='phone').text
print('\nPhone:')
print(business_phone)