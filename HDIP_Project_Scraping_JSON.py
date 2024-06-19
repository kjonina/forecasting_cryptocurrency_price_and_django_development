
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


## getting the range in yahoo finance
#url_yahoo_finance = []
#for num in range(0,1001,25):
#    url = 'https://finance.yahoo.com/cryptocurrencies/?offset=25&count='+ str(num)
#    url_yahoo_finance.append(url)

headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}

url = 'https://finance.yahoo.com/cryptocurrencies/'
# url = 'https://coinmarketcap.com/'

# getting the live page
def get_page_contents():
    response = requests.get(url , headers = headers)
    return response.content


# function to extract title
def get_title(soup):
    print(soup.title.string)

# function to parste html code
def convert_to_soup(content):
    return BeautifulSoup(content, features="html.parser")

# function to get pattern for JSON
def get_pattern(soup):
    global crypto_json
    
    pattern = re.compile(r'\s--\sData\s--\s')

    script_data = soup.find('script', text = pattern).contents[0]

    start = script_data.find("context")-2
    
    json_data = json.loads(script_data[start:-12])

    # examining the columns
#    columns_json = json_data['context']['dispatcher']['stores']['ScreenerResultsStore']['results']['columns']
    # Checking that it works
#    print('=====================================')
#    print('Columns\' Names')
#    print('=====================================')
#    print(columns_json)
    
    # this is where the data is
    crypto_json = json_data['context']['dispatcher']['stores']['ScreenerResultsStore']['results']['rows']
#   # Checking that it works    
#    print('=====================================')
#    print('Printing First JSON - Bitcoin')
#    print('=====================================')
#    print(crypto_json[0])

def get_df(json_data):
    
    global df_cryptolist
    df_cryptolist = pd.io.json.json_normalize(crypto_json)
    
    # creating a dataset with the right columns and correct column names
    df_cryptolist = pd.DataFrame({'Symbol': df_cryptolist['symbol'],
                   'Name': df_cryptolist['shortName'],
                   'Price (Intraday)': df_cryptolist['regularMarketPrice.fmt'],
                   'Change': df_cryptolist['regularMarketChange.fmt'],
                   '% Change': df_cryptolist['regularMarketChangePercent.fmt'],
                   'Market Cap': df_cryptolist['marketCap.fmt'],
                   'Volume in Currency (Since 0:00 UTC)': df_cryptolist['regularMarketVolume.fmt'],
                   'Volume in Currency (24Hr)': df_cryptolist['volume24Hr.fmt'],
                   'Total Volume All Currencies (24Hr)': df_cryptolist['volumeAllCurrencies.fmt'],
                   'Circulating Supply': df_cryptolist['circulatingSupply.fmt']})
#    # writing the dataset to csv
#    df_cryptolist.to_csv(r"df_cryptolist.csv", index =  False)
    
# Run live Website
def main():
    global df_cryptolist
    contents = get_page_contents()
    soup = convert_to_soup(contents)
    get_title(soup)
    get_pattern(soup)
    get_df(crypto_json)
    print(df_cryptolist.head(1).transpose())


if __name__ == '__main__':
    main()   