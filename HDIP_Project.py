"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Task:           Time Series Forecasting of Cryptocurrency
"""

# Downloading necessary files
import numpy as np
# from numpy import log
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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
import plotly.graph_objects as go
import re
import json
import requests 
import codecs
from bs4 import BeautifulSoup
import pandas as pd
from pandas.io.json import json_normalize


import statsmodels.api as sm
from pylab import rcParams

# importing functions from the file
from HDIP_Project_Functions import *


# display plotly in browser when run in Spyder
pio.renderers.default = 'browser'

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

from HDIP_Project_Functions import crypto_name, insert

# =============================================================================
# Collecting info from Yahoo Finance and creating a dataset for that cryptocurrency
# =============================================================================
create_df(insert)

create_y(insert)

from HDIP_Project_Functions import *

# =============================================================================
# Creating a graph examining the price and moving averages
# =============================================================================
price_sma_volume_chart()

candlestick_moving_average()

# =============================================================================
# Analysing the Histogram and Boxplot for crypto
# =============================================================================

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
# splitting the data 
create_train_and_test()

from HDIP_Project_Functions import *

# creating a plot for the training and test set
training_and_test_plot()

# =============================================================================
# 
# =============================================================================
create_diff_volume(y['diff'])

create_diff_log_diff()

# =============================================================================
# Techniques to remove Trend 
# =============================================================================
# Not Stationary
smoothing_with_moving_averages_method()

# Not Stationary
exponentian_with_moving_averages_method()

# STATIONARY - difference is the best method
differencing_method()

# =============================================================================
# Examining CLOSE
# =============================================================================

simple_seasonal_decompose(y['Close'], 365)
acf_and_pacf_plots(y['Close'])
KPSS_test(y['Close'])
adfuller_test(y['Close'])
rolling_mean_std(y['Close'], 365)

# =============================================================================
# Examining LOG CLOSE
# =============================================================================

simple_seasonal_decompose(y['log_Close'], 365)
acf_and_pacf_plots(y['log_Close'])
KPSS_test(y['log_Close'])
adfuller_test(y['log_Close'])
rolling_mean_std(y['log_Close'], 365)

# =============================================================================
# Examining DIFF - STATIONARY
# =============================================================================

simple_seasonal_decompose(y['diff'], 365)
acf_and_pacf_plots(y['diff'])
KPSS_test(y['diff'])
adfuller_test(y['diff'])
rolling_mean_std(y['diff'], 365)

# =============================================================================
# Examining LOG CLOSE DIFF - STATIONARY
# =============================================================================

simple_seasonal_decompose(y['log_Close_diff'], 365)
acf_and_pacf_plots(y['log_Close_diff'])
KPSS_test(y['log_Close_diff'])
adfuller_test(y['log_Close_diff'])
rolling_mean_std(y['log_Close_diff'], 365)



# =============================================================================
# Plotly 
# =============================================================================
decomposition(df['Close'], 365)

# =============================================================================
# Predicing and Forecasting the Closing Price with FBProphet 
# =============================================================================
# predicting price using FBProphet 
predict_prophet()
predict_prophet_components()
predict_prophet_plotly()

#Forecasting price using FBProphet 
forecast_prophet()
forecast_prophet_components()
forecast_prophet_plotly()



