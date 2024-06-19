"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Task:           Time Series Forecasting of Cryptocurrency
File:           This file is created to run functions with create dataframes, graphs, and test 
"""

# Downloading necessary files
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
import mplfinance as mpf 
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib import pyplot
import datetime
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.arima_model import ARMA
import plotly.graph_objects as go
import statsmodels.api as sm
from pylab import rcParams
import mpld3
from statsmodels.tsa.seasonal import seasonal_decompose
import re
import json
import requests 
import codecs
from bs4 import BeautifulSoup
import pandas as pd
from pandas.io.json import json_normalize
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
from math import sqrt



# display plotly in browser when run in Spyder
pio.renderers.default = 'browser'


# =============================================================================
# Getting Yahoo Table
# =============================================================================
# getting the live page
def get_yahoo_table():
    global df_cryptolist
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
    
# =============================================================================
# getting a list from the table
# =============================================================================
cryptolist = [] 
def get_crypto_df():
    index = 0
        
    while index < len(df_cryptolist.iloc[:,0]):
        try:
            for crypto in df_cryptolist.iloc[:,0]:
                cryptolist.append(str(crypto))
                index += 1
                
        except:
            index = len(df_cryptolist.iloc[:,0])
            break
    return cryptolist
          
# ============================================================================
# Trying to create an error message    
# ============================================================================
def please_choose_crypto():   
    global crypto_name
    global insert
    
    while True:
        print('============================================================')
        print('Top', len(df_cryptolist), 'Cryptocurrencies')
        print('============================================================')
        print(df_cryptolist[['Symbol','Name','Market Cap']].head(len(df_cryptolist)))
        try:
            insert = str(input('What cryptocurrency would you like to try out? Please select a symbol: ')).upper()
            #found = df_cryptolist[df_cryptolist['Symbol'].str.contains(insert)]
            crypto_name = str(df_cryptolist[df_cryptolist['Symbol'].str.contains(insert)].iloc[:,1]).split(' ')[4]
            
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        
        if not insert in cryptolist:
            print('Sorry. You did not select an available symbol or you misspelled the symbol')
                
        else:
            print('============================================================')
            print('You have selected: ', insert)
            df_new = df_cryptolist.copy()
            df_new.set_index("Symbol", inplace=True)
            df_new.head()
            print('============================================================')
            print(df_new.loc[insert])
            print('============================================================')
            break

# =============================================================================
# Collecting info from Yahoo Finance and creating a dataset for that cryptocurrency
# =============================================================================
def create_df(x):

    # =============================================================================
    # Creating a new dataset
    # =============================================================================
    
    global df
    
    start = "2009-01-01"
    end = dt.datetime.now()
    short_sma = 50
    long_sma = 200
    
    # creating a dataset for selected cryptocurrency 
    df = yf.download(x, start, end,interval = '1d')
    df = pd.DataFrame(df.dropna(), columns = ['Open', 'High','Low','Close', 'Adj Close', 'Volume'])


    # preparing data from time series analysis
    # eliminating any NAs - in most cryptocurrencies there are 4 days missing
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')

    print('============================================================')
    print('Nan in each columns before Backfill for' , crypto_name)
    print('------------------------------------------------------------')  
    print(df.isna().sum())
    print('The df has {} rows and {} columns.'.format(*df.shape))
    
    df = df.bfill()
    print('============================================================')
    print('Nan in each columns after Backfill for' , crypto_name)
    print('------------------------------------------------------------')  
    print(df.isna().sum())
    print('The df has {} rows and {} columns.'.format(*df.shape))
#    df = df.dropna()
    
    # Create short SMA
    df['short_SMA'] = df.iloc[:,1].rolling(window = short_sma).mean()
    
    # Create Long SMA
    df['long_SMA'] = df.iloc[:,1].rolling(window = long_sma).mean()
    
    # Create daily_return
    df['daily_return'] = df['Close'].pct_change(periods=1).mul(100)
    
    # Create monthly_return
    df['monthly_return'] = df['Close'].pct_change(periods=30).mul(100)
    
    # Create annual_return
    df['annual_return'] = df['Close'].pct_change(periods=365).mul(100)
    df['Name'] = crypto_name

    print('============================================================')
    print(crypto_name, '- Full Dataset')
    print('------------------------------------------------------------')
    print(df.head())
    print('------------------------------------------------------------')
    print(crypto_name, 'Full Dataset - Column Names')
    print(df.columns)

    print('============================================================')
    print('Nan in each columns after Backfill for' , crypto_name)
    print('------------------------------------------------------------')  
    print(df.isna().sum())
    print('============================================================')
    

#    # write to csv
#    df.to_csv(r"df.csv", index =  True)
    
    # =============================================================================
    # Assigning the target variable
    # =============================================================================

    
def create_y(x):
    
    global y
    
    y = pd.DataFrame(df['Close'], columns = ['Close'])
    y.sort_index(inplace = True)
    y['Name'] = crypto_name
    
    # examining the pct_change
    y['Close Percentage Change'] = y['Close'].pct_change(1)
    
    # Creating a new variable, examining the difference for each observation
    y['diff'] = y['Close'].diff()

    # logging the target varialbe due to great variance
    y['log_Close'] = np.log(y['Close'])
    
    # Creating a new variable, examining the difference for each observation
    y['log_Close_diff'] = y['log_Close'].diff()
    
    y['Logged Close Percentage Change'] = y['log_Close'].pct_change(1)
    
    # dropping the first na (because there is no difference)
    y = y.dropna()
    

#    # write to csv
#    y.to_csv(r"y.csv", index =  True)

    print('============================================================')
    print(crypto_name, '- Target Variable')
    print('------------------------------------------------------------')
    print(y.head())
    print('------------------------------------------------------------')
    print('Column Names')
    print(y.columns)
    print('============================================================')
    

# =============================================================================
# Creating a graph examining the price and moving averages
# =============================================================================
def price_sma_volume_chart():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[
            'Price and Death Cross of {}'.format(str(crypto_name)),
            'Volume of {}'.format(str(crypto_name))])

    # Lineplots of price and moving averages
    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['Close'],
                            name = crypto_name,
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Closing Price: %{y:$,.2f}<br>" ,
                            line = dict(color="black")), row = 1, col = 1)

    fig.add_trace(go.Scatter(x = df.index,
                             y = df['short_SMA'],
                             name = 'Short SMA 50-Day',
                             mode = 'lines',
                             customdata = df['Name'],
                             hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Short (50-Day) Moving Average Price: %{y:$,.2f}<br>",
                             line = dict(color="red")), row = 1, col = 1)

    fig.add_trace(go.Scatter(x = df.index,
                             y = df['long_SMA'],
                             name = 'Long SMA 200-Day',
                             mode = 'lines',
                             customdata = df['Name'],
                             hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Long (200-Day) Moving Average Price: %{y:$,.2f}<br>",
                             line = dict(color="green")), row = 1, col = 1)
    # Barplot of volume
    fig.add_trace(go.Bar(x = df.index,
                    y = df['Volume'],
                    name = 'Volume',
                    customdata = df['Name'],
                    hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Volume: %{y:,.}<br>" +
                                    "<extra></extra>",
                    marker = dict(color="black", opacity = True)), row = 2, col = 1)
    # Add title
    fig.update_layout(
            title = 'Summary of {}'.format(str(crypto_name)),
            title_font_size=30)
    
    
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='Volume'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.', row = 1, col = 1)
    #time buttons
    fig.update_xaxes(rangeselector= {'visible' :False}, row = 2, col = 1)

    #Show
    fig.show()



def candlestick_moving_average():

    fig  = go.Figure()

    trace1 = go.Candlestick(
        x = df.index,
        open = df["Open"],
        high = df["High"],
        low = df["Low"],
        close = df["Close"],
        name = crypto_name)

    data = [trace1]

    for i in range(5, 201, 5):

        sma = go.Scatter(
            x = df.index,
            y = df["Close"].rolling(i).mean(), # Pandas SMA
            name = "SMA" + str(i),
            line = dict(color = "#3E86AB",width=3),
            customdata = df['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Simple Moving Average Price: %{y:$,.2f}<br>",
            opacity = 0.7,
            visible = False,
        )

        data.append(sma)

    sliders = dict(

        # GENERAL
        steps = [],
        currentvalue = dict(
            font = dict(size = 16),
            prefix = "Simple Moving Average Step: ",
            xanchor = "left",
        ),

        x = 0,
        y = 0,
        len = 1,
        pad = dict(t = 0, b = 0),
        yanchor = "bottom",
        xanchor = "left",
    )

    for i in range((200 // 5) + 1):

        step = dict(
            method = "restyle",
            label = str(i * 5),
            value = str(i * 5),
            args = ["visible", [False] * ((200 // 5) + 1)],
        )

        step['args'][1][0] = True
        step['args'][1][i] = True
        sliders["steps"].append(step)



    layout = dict(

        title = 'Price of {}'.format(str(crypto_name),
            title_font_size=30),

        # ANIMATIONS
        sliders = [sliders],
        xaxis = dict(

            rangeselector = dict(
                activecolor = "#888888",
                bgcolor = "#DDDDDD",
                buttons = [
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD"),
                ]
            ),

        ),
        yaxis = dict(
            tickprefix = "$", tickformat = ',.',
            type = "linear",
            domain = [0.25, 1],
        ),

    )



    fig = go.Figure(data = data, layout = layout)


#
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_layout(showlegend=False)
    
    #Show
    fig.show()


# =============================================================================
# Analysing the Histogram and Boxplot for crypto
# =============================================================================

def create_hist_and_box(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Histogram of {} price'.format(crypto_name),
                                        'Box plot of {} price'.format(crypto_name)],
                        x_title = 'US Dollars')
    # 1.Histogram
    fig.add_trace(go.Histogram(x = data, name = 'Histogram', nbinsx = round(len(data) / 20),
#                               customdata = df['Name'],
#                               hovertemplate="<b>%{customdata}</b>"
                               ), row=1, col=1)
    
    #2. Boxplot 
    fig.add_trace(go.Box(x = data, name = 'Boxplot',
                         customdata = df['Name'],
                         hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Closing Price: %{x:$,.2f}<br>"+
                                    "<extra></extra>"), row=2, col=1)

    fig.update_layout(title = 'Plots of {} price'.format(crypto_name))
    fig.update_xaxes(tickprefix = '$', tickformat = ',.')
    fig.show()
    

# creating graph for Close Percentage Change
def create_hist_and_box_pct_change():
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Histogram of {} 1-Day Close Percentage Change'.format(crypto_name),
                                        'Box plot of {} 1-Day Close Percentage Change'.format(crypto_name)],
                        x_title = '1-Day Close Percentage Change')
    # 1.Histogram
    fig.add_trace(go.Histogram(x = y['Close Percentage Change'], name = 'Histogram', nbinsx = round(len(y) / 20),
                               ), row=1, col=1)

    #2. Boxplot
    fig.add_trace(go.Box(x = y['Close Percentage Change'], name = 'Boxplot',
                         customdata = y['Name'],
                         hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "1-Day Percentage Change: %{x:.0%}<br>"+
                                    "<extra></extra>"
                    ), row=2, col=1)

    fig.update_layout(title = 'Plots of 1-Day Close Percentage Change for {}'.format(crypto_name))
    fig['layout']['yaxis1']['title'] = '# of Observations'

    fig.update_xaxes(tickformat = '.0%', row = 1, col = 1)
    fig.update_xaxes(tickformat = '.0%', row = 2, col = 1)
    fig.update_layout(showlegend=False)

    fig.show()
    
    

def logged_create_hist_and_box_pct_change():
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Logged Closing Price - Histogram of {} 1-Day Close Percentage Change'.format(crypto_name),
                                        'Logged Closing Price - Box plot of {} 1-Day Close Percentage Change'.format(crypto_name)],
                        x_title = 'Loogged Price -  1-Day Close Percentage Change')
    # 1.Histogram
    fig.add_trace(go.Histogram(x = y['Logged Close Percentage Change'], name = 'Histogram', nbinsx = round(len(df) / 20),
                               ), row=1, col=1)
    
    #2. Boxplot 
    fig.add_trace(go.Box(x = y['Logged Close Percentage Change'], name = 'Boxplot',
                         customdata = df['Name'],
                         hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "1-Day Percentage Change: %{x:.0%}<br>"+
                                    "<extra></extra>"), row=2, col=1)

    fig.update_layout(title = 'Loogged Closing Price - Plots of 1-Day Close Percentage Change for {}'.format(crypto_name))
    fig['layout']['yaxis1']['title'] = '# of Observations'
    fig.update_xaxes(tickformat = '.0%', row = 1, col = 1)
    fig.update_xaxes(tickformat = '.0%', row = 2, col = 1)
    fig.show() 

# =============================================================================
# Creating a plot with analysis and rolling mean and standard deviation
# =============================================================================
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 365).mean()
    rolstd = timeseries.rolling(window = 365).std()

    #Plot rolling statistics:   
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = timeseries.index,
                                y = timeseries,
                                name = 'Original', 
                                mode='lines',
                                customdata = y['Name'],
                                hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Closing Price: %{y:$,.2f}<br>" +
                                    "<extra></extra>",
                                line = dict(color="blue")))
    fig.add_trace(go.Scatter(x = timeseries.index,
                                y = rolmean,
                                name = 'Rolling Mean', 
                                mode='lines',
                                customdata = y['Name'],
                                hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Rolling Mean Price: %{y:$,.2f}<br>" +
                                    "<extra></extra>",
                                line = dict(color="red")))
    fig.add_trace(go.Scatter(x = y.index,
                                y = rolstd,
                                name = 'Rolling Std', 
                                mode='lines',
                                customdata = y['Name'],
                                hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Rolling Std: %{y:$,.2f}<br>" +
                                    "<extra></extra>",
                                line = dict(color="black")))
    # Add titles
    fig.update_layout(
            title = 'Rolling Mean & Standard Deviation of {}'.format(crypto_name),
            yaxis_title = 'US Dollars',
            yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    #Show
    fig.show()


# =============================================================================
# Exploring the difference
# =============================================================================
# creating the plot to examine the difference
def diff_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data.index,
                            y = data,
                            name = str(crypto_name), 
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Price Volatility: %{y:$,.2f}<br>"+
                                    "<extra></extra>"))
    # Add titles
    fig.update_layout(
        title = 'Price of {}'.format(crypto_name),
        yaxis_title = 'US Dollars',
        yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    #Show
    fig.show()
    
# =============================================================================
# Diff and volume plot
# =============================================================================

def create_diff_volume(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=['Differnce of {} price'.format(crypto_name),
                                        'Volume of {}'.format(crypto_name)])
    # 1.Difference
    fig.add_trace(go.Scatter(x = data.index,
                            y = data,
                            name = str(crypto_name), 
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Price Volatility: %{y:$,.2f}<br>"+
                                    "<extra></extra>"), row = 1, col =1)
    #2. Volume
    # Barplot of volume 
    fig.add_trace(go.Bar(x = df.index,
                    y = df['Volume'],
                    name = 'Volume',
                    # corrects hovertemplate labels!
                    customdata = df['Name'],  
                    hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Volume: %{y:,.}<br>" +
                                    "<extra></extra>",
                    marker = dict(color="black", opacity = True)), row = 2, col = 1)
    # Add titles
    fig.update_layout( 
            title = 'Price of {}'.format(str(crypto_name)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='Volume'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.', row = 1, col = 1)
    #time buttons 
    fig.update_xaxes(rangeselector= {'visible' :False}, row = 2, col = 1)

    #Show
    fig.show()


# =============================================================================
# 
# =============================================================================

def create_diff_log_diff():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=['Difference of Closing {} Price'.format(crypto_name),
                                        'Logged Closing {} Price Difference'.format(crypto_name)])
    # 1.Difference
    fig.add_trace(go.Scatter(x = y.index,
                            y = y['diff'],
                            name = str(crypto_name), 
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Price Volatility: %{y:$,.2f}<br>"+
                                    "<extra></extra>"), row = 1, col =1)
    # 1.Difference of log
    fig.add_trace(go.Scatter(x = y.index,
                            y = y['log_Close_diff'],
                            name = str(crypto_name), 
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Logged Price Difference: %{y:,.2f}<br>"+
                                    "<extra></extra>"), row = 2, col =1)
    # Add titles
    fig.update_layout( 
            title = 'Price of {}'.format(str(crypto_name)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']=' '
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.', row = 1, col = 1)
    fig.update_xaxes(rangeselector= {'visible':False}, row=2, col=1)

    fig.update_xaxes(rangeslider= {'visible':False}, row=2, col=1)
    fig.update_layout(showlegend=False)

    fig.show()
    
# =============================================================================
#  daily, monthly, annual returns
# =============================================================================

def returns():
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, subplot_titles=[
            'Closing Price of {}'.format(str(crypto_name)),
            'Daily Return of {}'.format(str(crypto_name)),
            'Monthly Return of {}'.format(str(crypto_name)),
            'Annual Return of {}'.format(str(crypto_name))])
    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['Close'],
                            mode='lines',
                            customdata = df['Name'], name = 'Closing Price',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Closing Price: %{y:$,.2f}<br>"+
                                            "<extra></extra>"), row = 1, col = 1)

    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['daily_return'],
                            mode='lines',
                            customdata = df['Name'], name = 'Daily Return',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Daily Return: %{y:,.0%}<br>"+
                                            "<extra></extra>"), row = 2, col = 1)

    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['monthly_return'],
                            mode='lines',
                            customdata = df['Name'], name = 'Monthly Return',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Monthly Return: %{y:,.0%}<br>"+
                                            "<extra></extra>"), row = 3, col = 1)

    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['annual_return'],
                            mode='lines',
                            customdata = df['Name'], name = 'Annual Return',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Annual Return: %{y:,.0%}<br>"+
                                            "<extra></extra>"), row = 4, col = 1)

    # Add titles
    fig.update_layout(
            title = 'Price of {}'.format(str(crypto_name)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='% Return'
    fig['layout']['yaxis3']['title']='% Return'
    fig['layout']['yaxis4']['title']='% Return'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider= {'visible':False}, row=2, col=1)
    fig.update_xaxes(rangeslider= {'visible':False}, row=3, col=1)
    fig.update_xaxes(rangeslider= {'visible':False}, row=4, col=1)

    fig.update_xaxes(rangeselector= {'visible':False}, row=2, col=1)
    fig.update_xaxes(rangeselector= {'visible':False}, row=3, col=1)
    fig.update_xaxes(rangeselector= {'visible':False}, row=4, col=1)

    fig.update_yaxes(tickprefix = '$', tickformat = ',.', row = 1, col = 1)
    fig.update_yaxes(tickformat = ',.0%', row = 2, col = 1)
    fig.update_yaxes(tickformat = ',.0%', row = 3, col = 1)
    fig.update_yaxes(tickformat = ',.0%', row = 4, col = 1)

    fig.update_layout(showlegend=False)
    #Show
    fig.show()


# =============================================================================
# Splitting the data in Training and Test Data
# =============================================================================
##  Splitting based on 90:10 
#def create_train_and_test():
#    global df_train 
#    global df_test
#    global test_period
#    # Train data - 90%
#    df_train = y[:int(0.90*(len(y)))]
#    
#    print('============================================================')
#    print('{} Training Set'.format(crypto_name))
#    print('============================================================')
#    print(df_train.head())
#    print('Training set has {} rows and {} columns.'.format(*df_train.shape))
#    # Test data - 90%
#    df_test = y[int(0.90*(len(y))):]
#    test_period =  len(df_test)
#    print('============================================================')
#    print('{} Test Set'.format(crypto_name))
#    print('============================================================')
#    print(df_test.head())
#    print('Test set has {} rows and {} columns.'.format(*df_test.shape))
   
def create_train_and_test_period():
    
    global df_train 
    global df_test
    global test_period
    test_period= int(input('Please select the number for days to be used in the validation period: '))


    if test_period > 100:
        print('Please select validation period below 100 days')
    else:
        # Train data 
        df_train = y[:-test_period]
        print('============================================================')
        print('{} Training Set'.format(crypto_name))
        print('============================================================')
        print(df_train.head())
        print('Training set has {} rows and {} columns.'.format(*df_train.shape))
    
        # Test data 
        df_test = y[-test_period:]
        print('============================================================')
        print('{} Test Set'.format(crypto_name))
        print('============================================================')
        print(df_test.head())
        print('Test set has {} rows and {} columns.'.format(*df_test.shape))
        return df_train, df_test


def training_and_test_plot(): 
    # creating a plotly graph for training and test set
    trace1 = go.Scatter(
        x = df_train.index,
        y = df_train['Close'],
        customdata = df_train['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        name = 'Training Set')
    
    trace2 = go.Scatter(
        x = df_test.index,
        y = df_test['Close'],
        name = 'Test Set',
        customdata = df_test['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    fig.update_layout({'title': {'text':'Training and Test Set Plot {} for {} days'.format(crypto_name, test_period)}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()


# =============================================================================
# creating important functions for Time Series Analysis 
# =============================================================================
def normalise():
    # Select first prices
    first_price = df['Close'].iloc[0]
    # Create normalized
    normalized = df['Close'].div(first_price)
    # Plot normalized
    normalized.plot()
    plt.show()


# Dickey Fuller Test
def adfuller_test(data):
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print('============================================================')
    print('Results of Dickey-Fuller Test for {}:'.format(crypto_name))
    print('============================================================')
    print (dfoutput)
    if dftest[1]>0.05:
        print('Conclude not stationary')
    else:
        print('Conclude stationary')


def adfuller_test_for_Django(data, crypto_name):
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    dfoutput = pd.DataFrame(dfoutput)
    dfoutput = dfoutput.reset_index()
    dfoutput = dfoutput.rename(columns={'index': crypto_name, '0': 0})
    dfoutput1 = pd.DataFrame([['Stationary', np.where(dftest[1]>0.05, 'Conclude not stationary', 'Conclude stationary')]], columns=[crypto_name, 0])
    
    dfoutput = pd.concat([dfoutput,dfoutput1], sort=False).reset_index(drop=True)
    print(dfoutput)

    
# KPSS Test
def KPSS_test(data):
    result = kpss(data.values, regression='c', nlags='auto')
    print('============================================================')
    print('Results of KPSS Test for {}:'.format(crypto_name))
    print('============================================================')
    print('\nKPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[3].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

# seasonal decomposition
def simple_seasonal_decompose(data,number):
    rcParams['figure.figsize'] = 10, 8
    decomposition = seasonal_decompose(data, model='additive', period=number)
    decomposition.plot()
    plt.show()
    

def acf_and_pacf_plots(data):
    sns.set_style('darkgrid')
#    fig, (ax1, ax2,ax3) = plt.subplots(3,1, figsize = (8,15)) # graphs in a column
    fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize = (20,5)) # graphs in a row
    fig.suptitle('ACF and PACF plots of Logged Closing Price Difference for {}'.format(crypto_name), fontsize=16)
    ax1.plot(data)
    ax1.set_title('Original')
    plot_acf(data, lags=40, ax=ax2);
    plot_pacf(data, lags=40, ax=ax3);


def rolling_mean_std(timeseries, freq): 
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=freq).mean()
    rolstd = timeseries.rolling(window=freq).std()
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


# =============================================================================
# Monthly Data - 2511 observations to 82 - Not good
# =============================================================================
## RESAMPLING DATA INTO MONTHL1Y
#monthly_y = y.copy()
#monthly_y.resample('M').mean().head()
#monthly_y = monthly_y.asfreq('M')
##monthly_y.resample('M').median().head()
#
#
## DIFF - STATIONARY
#simple_seasonal_decompose(monthly_y['diff'], 12)
#acf_and_pacf_plots(monthly_y['diff'])
#KPSS_test(monthly_y['diff'])
#adfuller_test(monthly_y['diff'])
#rolling_mean_std(monthly_y['diff'], 365)
#
#
## LOGGED CLOSE DIFF - STATIONARY
#simple_seasonal_decompose(monthly_y['log_Close_diff'], 12)
#acf_and_pacf_plots(monthly_y['log_Close_diff'])
#KPSS_test(monthly_y['log_Close_diff'])
#adfuller_test(monthly_y['log_Close_diff'])
#rolling_mean_std(monthly_y['log_Close_diff'], 365)

## =============================================================================
## Boxplots of Returns with PLOTLY
## =============================================================================
#def box_year():
#    fig = go.Figure()
#
#    
#    fig.add_trace(go.Box(x = df.index.year, y = df['daily_return'],
#                         customdata = df['Name'],
#                            hovertemplate="<b>%{customdata}</b><br><br>" +
#                                    "Date: %{x|%d %b %Y} <br>" +
#                                    "Daily Return: %{y:.0%}<br>"+
#                                    "<extra></extra>"))
#
#    fig.update_layout(
#        title = 'Daily Returns of {}'.format(crypto_name),
#        yaxis_title = '% Change',
#    yaxis_tickformat = ',.0%')
#    fig.show()
#
#box_year()

# =============================================================================
# Techniques to remove Trend 
# =============================================================================

def smoothing_with_moving_averages_method():
    
    window =  int(input('What is your preferred number for the window? '))
    global ts_log
    global moving_avg
    global ts_log_moving_avg_diff
    
    ts_log = np.log(y['Close'])
    plt.plot(ts_log, color = 'green')
    moving_avg = ts_log.rolling(window=window).mean()
    plt.plot(moving_avg, color='red')
#    plt.legend(loc='best')
    plt.title('{}-Day Moving Average of Logged Closing Price'.format(window))
    plt.show(block=False)
 
    
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.head(15)
    ts_log_moving_avg_diff.dropna(inplace=True)
    
    #Determing rolling statistics
    rolmean = ts_log_moving_avg_diff.rolling(window=window).mean()
    rolstd = ts_log_moving_avg_diff.rolling(window=window).std()

    #Plot rolling statistics:
    orig = plt.plot(ts_log_moving_avg_diff, color='green',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('{}-Day Moving Average Difference of Log Closing Price vs Rolling Mean & Standard Deviation'.format(window))
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test of Moving Average Difference for Logged Closing Price:')
    dftest = adfuller(ts_log_moving_avg_diff, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
    
      
    if dftest[1]>0.05:
        print('Conclude not stationary')
    else:
        print('Conclude stationary') 

# Exponentially Weighted Moving Averages
def exponentially_weighted_moving_averages():
    
    window =  int(input('What is your preferred number for the window? '))
    expwighted_avg=ts_log.ewm(span=window).mean()
    plt.plot(ts_log, color = 'green')
    plt.plot(expwighted_avg, color='red')
    plt.legend(loc='best')
    plt.title('{}-Day Exponentially Weighted Moving Average of Logged Closing Price'.format(window))
    plt.show(block=False)
    plt.clf()
    
    ts_log.ewm(span=window).mean()
    ts_log_ewma_diff = ts_log - expwighted_avg

    #Determing rolling statistics
    rolmean = ts_log_ewma_diff.rolling(window=window).mean()
    rolstd = ts_log_ewma_diff.rolling(window=window).std()
    
    
    #Plot rolling statistics:
    orig = plt.plot(ts_log_ewma_diff, color='green',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('{}-Day Exponentially Weighted Moving Averages vs Rolling Mean & Standard Deviation'.format(window))
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test of Exponentially Weighted Moving Average Difference for Logged Closing Price:')
    dftest = adfuller(ts_log_ewma_diff, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
    if dftest[1]>0.05:
        print('Conclude not stationary')
    else:
        print('Conclude stationary') 



def differencing_method():
    window =  int(input('What is your preferred number for the window? '))
    
    ts_log_diff = ts_log - ts_log.shift()
    plt.plot(ts_log_diff, color = 'green')
    ts_log_diff.dropna(inplace=True)
  
    #Determing rolling statistics
    rolmean = ts_log_diff.rolling(window=window).mean()
    rolstd = ts_log_diff.rolling(window=window).std()
    
    
    #Plot rolling statistics:
    orig = plt.plot(ts_log_diff, color='green',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('{}-Day Rolling Mean & Standard Deviation of Log Closing Price Difference '.format(window))
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts_log_diff, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dftest[1]>0.05:
        print('Conclude not stationary')
    else:
        print('Conclude stationary')
        
# =============================================================================
# Decomposition with PLOTLY PACKAGE!
# =============================================================================

def decomposition(data, period):

    
    decomposition = sm.tsa.seasonal_decompose(data, period=period)

    #seasonality
    decomp_seasonal = decomposition.seasonal

    #trend
    decomp_trend = decomposition.trend

    #residual
    decomp_resid = decomposition.resid

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[
            'Price  of {}'.format(str(crypto_name)),
            'Trend values of {}'.format(str(crypto_name)),
            'Seasonal values of {}'.format(str(crypto_name)),
            'Residual values of {}'.format(str(crypto_name))])


    fig.add_trace(go.Scatter(x = df.index,
                            y = data,
                            name = crypto_name,
                            mode='lines'),row = 1, col = 1)


    fig.add_trace(go.Scatter(x = df.index,
                            y = decomp_trend,
                            name = 'Trend',
                            mode='lines'),row = 2, col = 1)


    fig.add_trace(go.Scatter(x = df.index,
                            y = decomp_seasonal,
                            name = 'Seasonality',
                            mode='lines'),row = 3, col = 1)

    fig.add_trace(go.Scatter(x = df.index,
                            y = decomp_resid,
                            name = 'Residual',
                            mode='lines'),row = 4, col = 1)

    # Add titles
    fig.update_layout(
            title = 'Decomposition of {} for {} days'.format(str(crypto_name),period))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='Trend'
    fig['layout']['yaxis3']['title']='Seasonality'
    fig['layout']['yaxis4']['title']='Residual'
    fig.update_layout(showlegend=False)

    fig.show()

# =============================================================================
# ARIMA MODELS
# =============================================================================
    
def run_ARIMA_model(p,d,q):
    global predictions_ARIMA
    # ARIMA fit model
    model = ARIMA(y.log_Close_diff, order = (p,q,q))
    model_fit = model.fit()
    
#    plt.plot(y.log_Close_diff)
#    plt.plot(model_fit.fittedvalues, color='red')
#    plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-y.log_Close_diff)**2))
#    plt.show()   

    trace1 = go.Scatter(
        x = y.index,
        y = y['log_Close_diff'],
        name = 'Observed')
    
    trace2 = go.Scatter(
        x = y.index,
        y = model_fit.fittedvalues,
        name = 'Predicted',
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    rss = np.nansum((predictions_ARIMA-y['log_Close_diff'])**2)
    
    fig.update_layout({'title': {'text':'RSS: %.4f'% np.nansum((model_fit.fittedvalues-y.log_Close_diff)**2)}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show() 
        
    predictions_ARIMA_diff = pd.Series(model_fit.fittedvalues, copy=True)
#    print (predictions_ARIMA_diff.head())  

    # Cumulative Sum to reverse differencing:
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#    print (predictions_ARIMA_diff_cumsum.head())


    # Adding 1st month value which was previously removed while differencing:
    predictions_ARIMA_log = pd.Series(y.log_Close.iloc[0], index=y.log_Close.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#    print(predictions_ARIMA_log.head())
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('=====================================')
    print('Predicting with ARIMA - {}'.format(str(crypto_name)))
    print('=====================================')
    print(predictions_ARIMA)
    
#    plt.plot(y.Close)
#    plt.plot(predictions_ARIMA)
#    plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions_ARIMA-y.log_Close_diff)**2)/len(y.log_Close_diff))) 

    trace1 = go.Scatter(
        x = y.index,
        y = y['Close'],
        customdata = y['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Observed Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        name = 'Observed')
    
    trace2 = go.Scatter(
        x = predictions_ARIMA.index,
        y = predictions_ARIMA,
        name = 'Predicted',
        customdata = y['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    rss = np.sqrt(np.nansum((predictions_ARIMA-y.log_Close_diff)**2)/len(y.log_Close_diff))
    
    fig.update_layout({'title': {'text':'{} Price Forecasting Estimation Using ARIMA, RSS: {:.4f}%'.format(str(crypto_name), rss)}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()
    
    

def run_ARIMA_model_user_prompted():
    
    p =  int(input('What is your preferred number of lags based on ACF? '))
    d =  int(input('What is your preferred number of differecing? '))
    q =  int(input('What is your preferred number of lags based on PACF?  '))
    global predictions_ARIMA
    # ARIMA fit model
    model = ARIMA(y.log_Close_diff, order = (p,d,q))
    model_fit = model.fit()
    
#    plt.plot(y.log_Close_diff)
#    plt.plot(model_fit.fittedvalues, color='red')
#    plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-y.log_Close_diff)**2))
#    plt.show()   

    trace1 = go.Scatter(
        x = y.index,
        y = y['log_Close_diff'],
        name = 'Observed')
    
    trace2 = go.Scatter(
        x = y.index,
        y = model_fit.fittedvalues,
        name = 'Predicted',
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    
    fig.update_layout({'title': {'text':'RSS: %.4f'% np.nansum((model_fit.fittedvalues-y.log_Close_diff)**2)}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()
        
    predictions_ARIMA_diff = pd.Series(model_fit.fittedvalues, copy=True)
#    print (predictions_ARIMA_diff.head())  

    # Cumulative Sum to reverse differencing:
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#    print (predictions_ARIMA_diff_cumsum.head())


    # Adding 1st month value which was previously removed while differencing:
    predictions_ARIMA_log = pd.Series(y.log_Close.iloc[0], index=y.log_Close.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#    print(predictions_ARIMA_log.head())
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('=====================================')
    print('Predicting with ARIMA - {}'.format(str(crypto_name)))
    print('=====================================')
    print(predictions_ARIMA)
    
#    plt.plot(y.Close)
#    plt.plot(predictions_ARIMA)
#    plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions_ARIMA-y.log_Close_diff)**2)/len(y.log_Close_diff))) 
    

    trace1 = go.Scatter(
        x = y.index,
        y = y['Close'],
        customdata = y['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Observed Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        name = 'Observed')
    
    trace2 = go.Scatter(
        x = predictions_ARIMA.index,
        y = predictions_ARIMA,
        name = 'Predicted',
        customdata = y['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    rss = np.sqrt(np.nansum((predictions_ARIMA-y.log_Close_diff)**2)/len(y.log_Close_diff))
    
    fig.update_layout({'title': {'text':'{} Price Forecasting Estimation Using ARIMA, RSS: {:.4f}%'.format(str(crypto_name), rss)}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()
    

# =============================================================================
# ARIMA Validation
# =============================================================================
def arima_validation(df_train, df_test, test_period):
    global fcast

    p =  int(input('What is your preferred number of lags based on ACF? '))
    d =  int(input('What is your preferred number of differecing? '))
    q =  int(input('What is your preferred number of lags based on PACF?  '))
    # Instantiate the model
    model =  ARIMA(df_train['Close'], order=(p,d,q))

    # Fit the model
    results = model.fit()


    # Print summary
    # print(results.summary())

#    start_index = df_test.index.min()
#    end_index = df_test.index.max()


    #Predictions
    forecast = results.get_forecast(steps=len(df_test), dynamic = True)

    # Confidence level of 90%
    fcast = forecast.summary_frame(alpha=0.10)
    # print('============================================================')
    # print('Forecast')
    # print('============================================================')
    # print(fcast.tail())

    # a plotly graph for training and test set
    df_train = go.Scatter(
        x = df_train.index,
        y = df_train['Close'],
        name = 'Training Set',
        customdata = df_train['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>")

    dt_test = go.Scatter(
        x = df_test.index,
        y = df_test['Close'],
        name = 'Test Set',
        customdata = df_test['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>",
        yaxis="y1")

    y_upper = fcast['mean_ci_upper']
    y_lower = fcast['mean_ci_lower']

    upper_band = go.Scatter(
        x=df_test.index,
        y=y_upper,
        line= dict(color='#57b88f'),
        name = 'Upper Band',
        customdata = df_test['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>")


    lower_band = go.Scatter(
        x = df_test.index,
        y = y_lower,
        line = dict(color='#57b88f'),
        name = 'Lower Band',
        customdata = df_test['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>",
        fill='tonexty'
        )


    mean = go.Scatter(
        x=df_test.index,
        y=fcast['mean'],
        name = 'Predicted',
        marker=dict(color='red', line=dict(width=3)),
        customdata = df_test['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>")



    data = [df_train, dt_test, upper_band, lower_band, mean]
    fig = go.Figure(data = data)
    fig.update_layout(showlegend=False)

    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    fig.update_layout(xaxis_rangeslider_visible = False)


    fig.update_layout(title = 'Predicting Closing Price of {} Using ARIMA for {} days'.format(crypto_name, test_period),
            title_font_size=30)
    fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',.')

    fig.show()
    

# =============================================================================
# ARIMA forecasting
# =============================================================================

def arima_forecast():
    forecasting_period = int(input('Please select the number for days to be used in the forecasting period: '))
    
    if forecasting_period >100:
        print('Please select forecsting period under 100 days')
        
    else:

        p =  int(input('What is your preferred number of lags based on ACF? '))
        d =  int(input('What is your preferred number of differecing? '))
        q =  int(input('Please select the number for lags based on PACF?  '))
    
        mod = ARIMA(df['Close'], order=(p,d,q))
        # Estimate the parameters
        res = mod.fit()
        # print(res.summary())
    
    
        # Forecasting out-of-sample
        forecast = res.get_forecast(steps=forecasting_period, dynamic = True)
    
        # # Confidence level of 90%
        # print('============================================================')
        # print('Forecast')
        # print('============================================================')
        # print(forecast.summary_frame(alpha=0.10).tail())
    
        # Construct the forecasts
        fcast = forecast.summary_frame()
    
        # print(fcast.index)
        y_upper = fcast['mean_ci_upper']
        y_lower = fcast['mean_ci_lower']
    
        # a plotly graph for training and test set
        actual = go.Scatter(
            x = df.index,
            y = df['Close'],
            customdata = df['Name'],name = 'Acutal Price',
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Closing Price: %{y:$,.2f}<br>")
    
    
        upper_band = go.Scatter(
            x=y_upper.index,
            y=y_upper,
            name = 'Upper Band',
            customdata = df['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Predicted Closing Price: %{y:$,.2f}<br>",
            line= dict(color='#57b88f')
            )
    
    
        lower_band = go.Scatter(
            x=y_lower.index,
            y= y_lower,
            name = 'Lower Band',
            line= dict(color='#57b88f'),
            customdata = df['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Predicted Closing Price: %{y:$,.2f}<br>",
            fill='tonexty',
    
            )
    
    
        mean = go.Scatter(
            x=fcast['mean_ci_upper'].index,
            y=fcast['mean'],name = 'Predicted',
            marker=dict(color='red', line=dict(width=3)),
            customdata = df['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Predicted Closing Price: %{y:$,.2f}<br>")
    
        data = [actual, upper_band, lower_band, mean]
        fig = go.Figure(data = data)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                                dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                                dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                                dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                                dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                                dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                                dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                                dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                                dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                                dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
        fig.update_layout(xaxis_rangeslider_visible = False)
    
        fig.update_layout(title = 'Forecasting Closing Price of {} Using ARIMA for {} days'.format(str(crypto_name), forecasting_period),
                title_font_size=30)
    
        fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    
        fig.show()

# =============================================================================
# 
# =============================================================================

def arima_forecast_with_log():
    
    forecasting_period = int(input('Please select the number for days to be used in the forecasting period: '))
    
    if forecasting_period >100:
        print('Please select forecsting period under 100 days')
        
    else:

        p =  int(input('What is your preferred number of lags based on ACF? '))
        d =  int(input('What is your preferred number of differecing? '))
        q =  int(input('Please select the number for lags based on PACF?  '))
    
        mod = ARIMA(y['log_Close_diff'], order=(p,d,q))
        # Estimate the parameters
        res = mod.fit()
        # print(res.summary())
    
        # Forecasting out-of-sample
        forecast = res.get_forecast(steps=forecasting_period, dynamic = True)
        
        # # Confidence level of 90%
        # print('============================================================')
        # print('Forecast')
        # print('============================================================')
        # print(forecast.summary_frame(alpha=0.10).tail())
    
        # Construct the forecasts with Confidence level of 90%
        fcast = forecast.summary_frame(alpha=0.10)
        y_mean= fcast['mean_se']
    
        predictions_ARIMA_diff = pd.Series(y_mean, copy=True)
    #    print (predictions_ARIMA_diff.head())  
    
        # Cumulative Sum to reverse differencing:
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        print('=====================================')
        print('predictions_ARIMA_diff_cumsum- {}'.format(str(crypto_name)))
        print('=====================================')
        print(predictions_ARIMA_diff_cumsum.tail())
        
    
        # Adding 1st month value which was previously removed while differencing:
        predictions_ARIMA_log = pd.Series(y.log_Close.iloc[-1], index=fcast.index)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        print('=====================================')
        print('predictions_ARIMA_log- {}'.format(str(crypto_name)))
        print('=====================================')
        print(predictions_ARIMA_log.tail())
        
        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        print('=====================================')
        print('predictions_ARIMA - {}'.format(str(crypto_name)))
        print('=====================================')
        print(predictions_ARIMA.tail())
        
    
        # a plotly graph for training and test set
        actual = go.Scatter(
            x = df.index,
            y = df['Close'],
            customdata = df['Name'],name = 'Acutal Price',
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Closing Price: %{y:$,.2f}<br>")
    
    
    #    upper_band = go.Scatter(
    #        x=y_upper.index,
    #        y=y_upper,
    #        name = 'Upper Band',
    #        customdata = df['Name'],
    #        hovertemplate="<b>%{customdata}</b><br><br>" +
    #        "Date: %{x|%d %b %Y} <br>" +
    #        "Predicted Closing Price: %{y:$,.2f}<br>",
    #        line= dict(color='#57b88f')
    #        )
    #
    #
    #    lower_band = go.Scatter(
    #        x=y_lower.index,
    #        y= y_lower,
    #        name = 'Lower Band',
    #        line= dict(color='#57b88f'),
    #        customdata = df['Name'],
    #        hovertemplate="<b>%{customdata}</b><br><br>" +
    #        "Date: %{x|%d %b %Y} <br>" +
    #        "Predicted Closing Price: %{y:$,.2f}<br>",
    #        fill='tonexty',
    #        )
    
    
        mean = go.Scatter(
            x=predictions_ARIMA.index,
            y=predictions_ARIMA,
            name = 'Predicted',
            marker=dict(color='red', line=dict(width=3)),
            customdata = df['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Predicted Closing Price: %{y:$,.2f}<br>")
    
        data = [actual, mean#, upper_band, lower_band
                ]
        fig = go.Figure(data = data)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                                dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                                dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                                dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                                dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                                dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                                dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                                dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                                dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                                dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
        fig.update_layout(xaxis_rangeslider_visible = False)
    
        fig.update_layout(title = 'Forecasting Closing Price of {} Using ARIMA for {} days'.format(str(crypto_name), forecasting_period),
                title_font_size=30)
    
        fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    
        fig.show()




# =============================================================================
# 
# =============================================================================


def arima_validation_with_log(df_train,df_test):
    
    forecasting_period = int(input('Please select the number for days to be used in the forecasting period: '))
    
    if forecasting_period >100:
        print('Please select forecsting period under 100 days')
        
    else:

        p =  int(input('What is your preferred number of lags based on ACF? '))
        d =  int(input('What is your preferred number of differecing? '))
        q =  int(input('Please select the number for lags based on PACF?  '))
    
        mod = ARIMA(df_train['log_Close_diff'], order=(p,d,q))
        # Estimate the parameters
        res = mod.fit()
        # print(res.summary())
    
        # Forecasting out-of-sample
        forecast = res.get_forecast(steps=len(df_test), dynamic = True)
        
        # # Confidence level of 90%
        # print('============================================================')
        # print('Forecast')
        # print('============================================================')
        # print(forecast.summary_frame(alpha=0.10).tail())
    
        # Confidence level of 90%
        fcast = forecast.summary_frame(alpha=0.10)
        y_mean= fcast['mean_se']
    
        predictions_ARIMA_diff = pd.Series(y_mean, copy=True)
    #    print (predictions_ARIMA_diff.head())  
    
        # Cumulative Sum to reverse differencing:
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        print('=====================================')
        print('predictions_ARIMA_diff_cumsum- {}'.format(str(crypto_name)))
        print('=====================================')
        print(predictions_ARIMA_diff_cumsum.tail())
        
    
        # Adding 1st month value which was previously removed while differencing:
        predictions_ARIMA_log = pd.Series(y.log_Close.iloc[-1], index=fcast.index)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        print('=====================================')
        print('predictions_ARIMA_log- {}'.format(str(crypto_name)))
        print('=====================================')
        print(predictions_ARIMA_log.tail())
        
        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        print('=====================================')
        print('predictions_ARIMA - {}'.format(str(crypto_name)))
        print('=====================================')
        print(predictions_ARIMA.tail())


        # a plotly graph for training and test set
        df_train_trace = go.Scatter(
            x = df_train.index,
            y = df_train['Close'],
            name = 'Training Set',
            customdata = df_train['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Closing Price: %{y:$,.2f}<br>")
    
        df_test_trace = go.Scatter(
            x = df_test.index,
            y = df_test['Close'],
            name = 'Test Set',
            customdata = df_test['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Closing Price: %{y:$,.2f}<br>",
            yaxis="y1")
        
    
    #    upper_band = go.Scatter(
    #        x=y_upper.index,
    #        y=y_upper,
    #        name = 'Upper Band',
    #        customdata = df['Name'],
    #        hovertemplate="<b>%{customdata}</b><br><br>" +
    #        "Date: %{x|%d %b %Y} <br>" +
    #        "Predicted Closing Price: %{y:$,.2f}<br>",
    #        line= dict(color='#57b88f')
    #        )
    #
    #
    #    lower_band = go.Scatter(
    #        x=y_lower.index,
    #        y= y_lower,
    #        name = 'Lower Band',
    #        line= dict(color='#57b88f'),
    #        customdata = df['Name'],
    #        hovertemplate="<b>%{customdata}</b><br><br>" +
    #        "Date: %{x|%d %b %Y} <br>" +
    #        "Predicted Closing Price: %{y:$,.2f}<br>",
    #        fill='tonexty',
    #        )
    
    
        mean_trace = go.Scatter(
            x=predictions_ARIMA.index,
            y=predictions_ARIMA,
            name = 'Predicted',
            marker=dict(color='red', line=dict(width=3)),
            customdata = df_test['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
            "Date: %{x|%d %b %Y} <br>" +
            "Predicted Closing Price: %{y:$,.2f}<br>")
    
        data = [df_train_trace, df_test_trace, mean_trace]
        fig = go.Figure(data = data)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                                dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                                dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                                dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                                dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                                dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                                dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                                dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                                dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                                dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
        fig.update_layout(xaxis_rangeslider_visible = False)
    
        fig.update_layout(title = 'Forecasting Closing Price of {} Using ARIMA for {} days'.format(str(crypto_name), forecasting_period),
                title_font_size=30)
    
        fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    
        fig.show()

# =============================================================================
# Model Evaluation
# =============================================================================
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast(y,pred):
    global results
    results = pd.DataFrame({'r2_score':r2_score(y, pred),
                           }, index=[0])
    results['mean_absolute_error'] = '{:.4f}'.format(mean_absolute_error(y, pred))
    results['median_absolute_error'] = '{:.4f}'.format(median_absolute_error(y, pred))
    results['mse'] = '{:.4f}'.format(mean_squared_error(y, pred))
    results['msle'] = '{:.4f}'.format(mean_squared_log_error(y, pred))
    results['mape'] = '{:.4f}'.format(mean_absolute_percentage_error(y, pred))
    results['rmse'] = '{:.4f}'.format(np.sqrt(float(results['mse'])))

    results = pd.DataFrame(results).transpose()
    results = results.reset_index()
    print('=====================================')
    print('Model Evaluation - {}'.format(str(crypto_name)))
    print('=====================================')
    print(results)
    
# =============================================================================
# Predict Closing Price using FBProphet
# =============================================================================
def predict_prophet():
    global df_forecast
    global crypto
    global df_prophet
    
    crypto = df_train[['Close', 'Name']]
    crypto = crypto.reset_index()
    crypto = crypto.rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet = Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
    df_prophet.fit(crypto)
    
    df_forecast = df_prophet.make_future_dataframe(periods= len(df_test), freq='D')
    
    df_forecast = df_prophet.predict(df_forecast)
    df_forecast['Name'] = crypto['Name']
    
def predict_prophet_components():
    df_prophet.plot_components(df_forecast)


    
def predict_prophet_plotly(df_test, df_train, test_period):
    
    df_forecast['Name'] = df_forecast['Name'].replace(np.nan, crypto_name)


    df_train = go.Scatter(
        x = df_train.index,
        y = df_train['Close'],
        customdata = df_train['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>",
        name = 'Training Set')

    df_test = go.Scatter(
        x = df_test.index,
        y = df_test['Close'],
        name = 'Test Set',
        customdata = df_test['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>",
        yaxis="y1")

    trend = go.Scatter(
        name = 'Trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        customdata = df_forecast['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Trend: %{y:$,.2f}<br>",
        marker=dict(color='red', line=dict(width=3))
    )
    upper_band = go.Scatter(
        name = 'Upper Band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        customdata = df_forecast['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Upper Band: %{y:$,.2f}<br>",
        line= dict(color='#57b88f'),
        fill = 'tonexty'
    )
    lower_band = go.Scatter(
        name= 'Lower Band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        customdata = df_forecast['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Lower Band: %{y:$,.2f}<br>",
        line= dict(color='#57b88f')
       )


    data = [df_train, df_test, trend, lower_band, upper_band]

    layout = dict(title='Predicting Closing Price of {} Using FbProphet'.format(test_period, crypto_name),
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

    fig = go.Figure(data = data, layout=layout)
#    fig['layout']['yaxis1']['title']='US Dollars'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.')


    fig.update_layout(showlegend=False)
    
    fig.show()


# =============================================================================
# Forecasting Price with Prophet 
# =============================================================================
def forecast_prophet():
    global df_forecast
    global crypto
    global df_prophet
    
    crypto = df[['Close', 'Name']]
    crypto = crypto.reset_index()
    crypto = crypto.rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet = Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
    df_prophet.fit(crypto)
    
    estimated_days=int(input('Please select the number for days to forecast: '))
    df_forecast = df_prophet.make_future_dataframe(periods= estimated_days, freq='D')
    
    df_forecast = df_prophet.predict(df_forecast)
    df_forecast['Name'] = crypto['Name']
    
def forecast_prophet_components():
    df_prophet.plot_components(df_forecast)



def prophet_forecast():

    df_forecast['Name'] = df_forecast['Name'].replace(np.nan, crypto_name)

    actual = go.Scatter(
        x = df.index,
        y = df['Close'],
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>",
        name = 'Actual Price',
        marker = dict(line = dict(width=1))
        )

    trend = go.Scatter(
        name = 'Trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        customdata = df_forecast['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Trend: %{y:$,.2f}<br>",
        marker=dict(color='red', line=dict(width=3))
    )

    upper_band = go.Scatter(
        name = 'Upper Band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        customdata = df_forecast['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Upper Band: %{y:$,.2f}<br>",
        line= dict(color='#57b88f'),
        fill = 'tonexty'
    )

    lower_band = go.Scatter(
        name= 'Lower Band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        customdata = df_forecast['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Lower Band: %{y:$,.2f}<br>",
        line= dict(color='#57b88f')
       )

    data = [trend, lower_band, upper_band, actual]

    layout = dict(title='Forecasting Closing Price of {} Using FbProphet'.format(crypto_name),
                title_font_size=30, xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

    fig = go.Figure(data = data, layout=layout)
#    fig['layout']['yaxis1']['title']='US Dollars'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.')

    fig.update_layout(showlegend=False)

    fig.show()




# =============================================================================
# Getting the Yahoo Table with Beautiful Soup
# =============================================================================
get_yahoo_table()

# =============================================================================
# creating a list from the crypto-table
# =============================================================================      
get_crypto_df()
     
# ============================================================================
# Asking the user for an input   
# ============================================================================
please_choose_crypto()

# =============================================================================
# Collecting info from Yahoo Finance and creating a dataset for that cryptocurrency
# =============================================================================
create_df(insert)

create_y(insert)

# =============================================================================
# Creating a graph examining the price and moving averages
# =============================================================================
price_sma_volume_chart()

candlestick_moving_average()

# =============================================================================
# Analysing the Histogram and Boxplot for crypto
# =============================================================================

##considered useless
#create_hist_and_box(y['diff'])

create_hist_and_box_pct_change()

logged_create_hist_and_box_pct_change()

# =============================================================================
# Creating a plot with analysis and rolling mean and standard deviation
# =============================================================================
test_stationarity(df['Close'])

test_stationarity(y['Close Percentage Change'])

# =============================================================================
# Splitting the data in Training and Test Data
# =============================================================================
## splitting the data 
#create_train_and_test()

create_train_and_test_period()

# creating a plot for the training and test set
training_and_test_plot()

# =============================================================================
# Examining Differences
# =============================================================================

diff_plot(y['diff'])

create_diff_volume(y['diff'])

create_diff_log_diff()


# =============================================================================
# Returns
# =============================================================================
returns()


# =============================================================================
# Techniques to remove Trend 
# =============================================================================
# Try 30 and 365
smoothing_with_moving_averages_method()

# Try 30 and 365
exponentially_weighted_moving_averages()

# Try 30 and 365
differencing_method()


## =============================================================================
## Examining CLOSE
## =============================================================================
#
#simple_seasonal_decompose(y['Close'], 365)
#acf_and_pacf_plots(y['Close'])
#KPSS_test(y['Close'])
#adfuller_test(y['Close'])
#rolling_mean_std(y['Close'], 365)
#
## =============================================================================
## Examining LOG CLOSE
## =============================================================================
#
#simple_seasonal_decompose(y['log_Close'], 365)
#acf_and_pacf_plots(y['log_Close'])
#KPSS_test(y['log_Close'])
#adfuller_test(y['log_Close'])
#rolling_mean_std(y['log_Close'], 365)
#
## =============================================================================
## Examining DIFF - STATIONARY
## =============================================================================
#
#simple_seasonal_decompose(y['diff'], 365)
#acf_and_pacf_plots(y['diff'])
#KPSS_test(y['diff'])
#adfuller_test(y['diff'])
#rolling_mean_std(y['diff'], 365)

# =============================================================================
# Examining LOG CLOSE DIFF - STATIONARY
# =============================================================================

simple_seasonal_decompose(y['log_Close_diff'], 365)
acf_and_pacf_plots(y['log_Close_diff'])
KPSS_test(y['log_Close_diff'])
adfuller_test(y['log_Close_diff'])
rolling_mean_std(y['log_Close_diff'], 365)


# =============================================================================
# Decomposition 
# =============================================================================
decomposition(df['Close'], 365)

# =============================================================================
# ARIMA Models
# =============================================================================
## AR Model + evaluation
#run_ARIMA_model(1, 0, 0)
#evaluate_forecast(y.Close,predictions_ARIMA)
#
## AR + I Model + evaluation
#run_ARIMA_model(1, 1, 0)
#evaluate_forecast(y.Close,predictions_ARIMA)
#
## ARIMA Model + evaluation
#run_ARIMA_model(1, 1, 1)
#evaluate_forecast(y.Close,predictions_ARIMA)
#
## MA Model + evaluation
#run_ARIMA_model(0, 0, 1)
#evaluate_forecast(y.Close,predictions_ARIMA)
#
##AR + I model  + evaluation
#run_ARIMA_model(1, 1, 0)
#evaluate_forecast(y.Close,predictions_ARIMA)
    

# prompting the user to put in the p,q,d themselves  + evaluation
run_ARIMA_model_user_prompted()
evaluate_forecast(y.Close,predictions_ARIMA)


# =============================================================================
# Arima Validation and Evaluation
# =============================================================================
arima_validation(df_train, df_test, test_period)
evaluate_forecast(df_test['Close'],fcast['mean_se'])

# =============================================================================
# Arima Forecast
# =============================================================================
arima_forecast()

# something goes wrong when log is used
arima_validation_with_log(df_train,df_test)

arima_forecast_with_log()

# =============================================================================
# Prophet Validation and Evaluation
# =============================================================================
# predicting price using FBProphet 
predict_prophet()
predict_prophet_components()
predict_prophet_plotly(df_test, df_train, test_period)

# =============================================================================
# Prophet Forecast
# =============================================================================
#Forecasting price using FBProphet 
forecast_prophet()
forecast_prophet_components()
prophet_forecast()





