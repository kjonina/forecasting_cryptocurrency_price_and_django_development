"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Project Objective:           Time Series Forecasting of Cryptocurrency

Task: Scraping Yahoo Finance so that the user can select the crypto currency 
      based on Market Cap
"""
#importing important packages
import re
import json
#from io import String10
import requests 
import codecs
from bs4 import BeautifulSoup
import pandas as pd
from pandas.io.json import json_normalize


url_yahoo_finance = []
for num in range(0,51,25):
    url = 'https://finance.yahoo.com/cryptocurrencies/?offset=25&count='+ str(num)
    url_yahoo_finance.append(url)


# getting the live page
def get_yahoo_table():

    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    
    url = 'https://finance.yahoo.com/cryptocurrencies/'
    # url = 'https://coinmarketcap.com/'
    response = requests.get(url , headers = headers)
    content = response.content
    soup = BeautifulSoup(content, features="html.parser")
    pattern = re.compile(r'\s--\sData\s--\s')
    script_data = soup.find('script', text = pattern).contents[0]
    start = script_data.find("context")-2

    json_data = json.loads(script_data[start:-12])
    
    # this is where the data is
    crypto_json = json_data['context']['dispatcher']['stores']['ScreenerResultsStore']['results']['rows']

    return crypto_json[0]
    
get_yahoo_table() 