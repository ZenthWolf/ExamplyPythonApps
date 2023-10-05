#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:55:12 2023

@author: zenth
"""

#%% Cell Imports

import requests
from bs4 import BeautifulSoup

import os
current_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_directory)

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

featured_testimonial = soup.find_all('div', class_="quote")
for testimonial in featured_testimonial:
  print(testimonial.text)
  
staff = soup.find_all('div', class_="info col-xs-8 col-xs-offset-2 col-sm-7 col-sm-offset-0 col-md-6 col-lg-8")
for s in staff:
    print(s.text)

links = soup.find_all("a")

for link in links:
  print(link.text, link.get('href'))
#%% Recording

with open("Wisdom_Vet.txt", "w") as f:
    f.write(soup.prettify())

with open("Wisdom_Vet_Services.txt", "w") as f:
    for service in services_list:
        f.write(service.text)

