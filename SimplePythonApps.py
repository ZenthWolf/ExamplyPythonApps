#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:55:12 2023

@author: zenth
"""

#%% Cell Imports

import requests

#%% Scrape WisdomPetMedicine (fictional business)

response = requests.get('https://wisdompetmed.com')

response.url           # Check URL is correct
response.status_code   # Check website Status is good 2xx
response.headers       # Check headers
response.content       # HTML in byte data
response.text          # HTML as UTF-8 string (More readable, might less accurate)

