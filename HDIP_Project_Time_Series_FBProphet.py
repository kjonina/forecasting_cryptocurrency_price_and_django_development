# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:45:01 2021

@author: Karina
"""

# Downloading necessary Packages
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import mplfinance as mpf 
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib import pyplot
import datetime
from datetime import datetime
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf, adfuller, kpss
from statsmodels.tsa.arima_model import ARMA
import plotly.graph_objects as go
import statsmodels.api as sm
from pylab import rcParams
import statsmodels.api as sm

from fbprophet import Prophet

from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# display plotly in browser when run in Spyder
pio.renderers.default = 'browser'

# =============================================================================
# Read in file for ease of use
# =============================================================================

# read the CSV file
df_cryptolist = pd.read_csv('df_cryptolist.csv')

# read the CSV file
df = pd.read_csv('df.csv', parse_dates=['Date'], index_col='Date')

# read the CSV file
y = pd.read_csv('y.csv', parse_dates=['Date'], index_col='Date')

crypto_name = 'Ethereum'

insert = 'ETH-USD'


# Fixing issues with frequency

df = df.asfreq('D')
print('Nan in each columns' , df.isna().sum())


y = y.asfreq('D')
print('Nan in each columns' , y.isna().sum())


# =============================================================================
# Creating Train and Test Dataset
# =============================================================================
def create_train_and_test():
    global df_train 
    global df_test
    # Train data - 80%
    df_train = y[:int(0.90*(len(y)))]
    
    print('============================================================')
    print('{} Training Set'.format(crypto_name))
    print('============================================================')
    print(df_train.head())
    print('Training set has {} rows and {} columns.'.format(*df_train.shape))
    # Test data - 20%
    df_test = y[int(0.90*(len(y))):]
    print('============================================================')
    print('{} Test Set'.format(crypto_name))
    print('============================================================')
    print(df_test.head())
    print('Test set has {} rows and {} columns.'.format(*df_test.shape))
    
    
create_train_and_test()
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
    df_forecast['Name'] = crypto[['Name']]
    df_forecast['Name'] = df_forecast['Name'].replace(np.nan, crypto_name)


def predict_prophet_components():
    df_prophet.plot_components(df_forecast)


    
def predict_prophet_plotly():
    
    trace1 = go.Scatter(
        x = df_train.index,
        y = df_train['Close'],
        customdata = df_train['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>",
        name = 'Training Set')
    
    trace2 = go.Scatter(
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
        marker=dict(
            color='red',
            line=dict(width=3)
        )
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
       
       
    data = [trace1, trace2, trend, lower_band, upper_band]
    
    layout = dict(title='{} Price Forecasting Estimation Using FbProphet'.format(crypto_name),
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

    fig = go.Figure(data = data, layout=layout)
#    fig['layout']['yaxis1']['title']='US Dollars'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count = 7, label = "1W", step = "day", stepmode = "backward"),
                dict(count = 28, label = "1M", step = "day", stepmode = "backward"),
                dict(count = 6, label = "6M", step = "month", stepmode = "backward"),
                dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                dict(count = 1, label = "1Y", step = "year", stepmode = "backward"),
                dict(count = 3, label = "3Y", step = "year", stepmode = "backward"),
                dict(count = 5, label = "5Y", step = "year", stepmode = "backward"),
                dict(step = "all")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.')
    

    fig.update_layout(showlegend=False)  
    fig.show()




predict_prophet()
predict_prophet_components()
predict_prophet_plotly()



def all_predict_prophet():
    
    crypto = df_train[['Close']]
    crypto = crypto.reset_index()
    crypto = crypto.rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet = Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
    df_prophet.fit(crypto)
    
    df_forecast = df_prophet.make_future_dataframe(periods= len(df_test), freq='D')
    
    df_forecast = df_prophet.predict(df_forecast)
    df_forecast['Name'] = crypto[['Name']]
    df_forecast['Name'] = df_forecast['Name'].replace(np.nan, crypto_name)

    
    trace1 = go.Scatter(
        x = df_train.index,
        y = df_train['Close'],
        customdata = df_train['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>",
        name = 'Training Set')
    
    trace2 = go.Scatter(
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
        marker=dict(
            color='red',
            line=dict(width=3)
        )
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
       
       
    data = [trace1, trace2, trend, lower_band, upper_band]
    
    layout = dict(title='{} Price Forecasting Estimation Using FbProphet'.format(crypto_name),
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

    fig = go.Figure(data = data, layout=layout)
#    fig['layout']['yaxis1']['title']='US Dollars'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count = 7, label = "1W", step = "day", stepmode = "backward"),
                dict(count = 28, label = "1M", step = "day", stepmode = "backward"),
                dict(count = 6, label = "6M", step = "month", stepmode = "backward"),
                dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                dict(count = 1, label = "1Y", step = "year", stepmode = "backward"),
                dict(count = 3, label = "3Y", step = "year", stepmode = "backward"),
                dict(count = 5, label = "5Y", step = "year", stepmode = "backward"),
                dict(step = "all")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.')
    

    fig.update_layout(showlegend=False)  
    fig.show()


all_predict_prophet()


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
    
    estimated_days=91
    df_forecast = df_prophet.make_future_dataframe(periods= estimated_days*2, freq='D')
    
    df_forecast = df_prophet.predict(df_forecast)
    df_forecast['Name'] = crypto[['Name']]
    df_forecast['Name'] = df_forecast['Name'].replace(np.nan, crypto_name)
    
def forecast_prophet_components():
    df_prophet.plot_components(df_forecast)


def forecast_prophet_plotly():

    trace = go.Scatter(
        x = df.index,
        y = df['Close'],
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>",
        name = 'Actual Closing Price',
        marker = dict(
            color = 'blue',
            line = dict(width=3)
        ))
    
    trace1 = go.Scatter(
        name = 'Trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        customdata = df_forecast['Name'], 
        hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Trend: %{y:$,.2f}<br>",
        marker=dict(
            color='red',
            line=dict(width=3)
        )
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
       
       
    data = [trace1, lower_band, upper_band, trace]
    
    layout = dict(title='{} Price Forecasting Estimation Using FbProphet'.format(crypto_name),
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

    fig = go.Figure(data = data, layout=layout)
#    fig['layout']['yaxis1']['title']='US Dollars'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count = 7, label = "1W", step = "day", stepmode = "backward"),
                dict(count = 28, label = "1M", step = "day", stepmode = "backward"),
                dict(count = 6, label = "6M", step = "month", stepmode = "backward"),
                dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                dict(count = 1, label = "1Y", step = "year", stepmode = "backward"),
                dict(count = 3, label = "3Y", step = "year", stepmode = "backward"),
                dict(count = 5, label = "5Y", step = "year", stepmode = "backward"),
                dict(step = "all")])))
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.')
    

    fig.update_layout(showlegend=False)  
    fig.show()



forecast_prophet()
forecast_prophet_components()
forecast_prophet_plotly()

# =============================================================================
# Evaluation
# =============================================================================

def prophet_evaluation():
    
    df_forecast['dtest_trend'] = df_forecast['trend'].iloc[-len(df_test):]

    df_forecast1= df_forecast[['dtest_trend']].dropna()
    results = pd.DataFrame({'R2 Score':r2_score(df_test['Close'], df_forecast1['dtest_trend']),
                            }, index=[0])
    results['Mean Absolute Error'] = '{:.4f}'.format(np.mean(np.abs((df_test['Close'] - df_forecast1['dtest_trend']) / df_test['Close'])) * 100)
    results['Median Absolute Error'] = '{:.4f}'.format(median_absolute_error(df_test['Close'], df_forecast1['dtest_trend']))
    results['MSE'] = '{:.4f}'.format(mean_squared_error(df_test['Close'], df_forecast1['dtest_trend']))
    results['MSLE'] = '{:.4f}'.format(mean_squared_log_error(df_test['Close'], df_forecast1['dtest_trend']))
    results['MAPE'] = '{:.4f}'.format(np.mean(np.abs((df_test['Close'] - df_forecast1['dtest_trend']) / df_test['Close'])) * 100)
    results['RMSE'] = '{:.4f}'.format(np.sqrt(float(results['MSE'])))
    
    results = pd.DataFrame(results).transpose()
    results = results.reset_index()
    return results.to_json(orient='records')


prophet_evaluation()