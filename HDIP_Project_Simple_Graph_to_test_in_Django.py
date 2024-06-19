
"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Task:           Time Series Forecasting of Cryptocurrency

Aim:            This file was used to create a very simple graph and to print it in Django
"""

# Downloading necessary files
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


#https://stackoverflow.com/questions/36846395/embedding-a-plotly-chart-in-a-django-template


# display plotly in browser when run in Spyder
pio.renderers.default = 'browser'

df = pd.read_csv('df.csv')


# used this code to get the arrays
# Closing price in the last 50 days
np.array(df['Close'].head(50))

# Date for the last 50 days
np.array(df['Date'].head(50))

# Volume for the last 50 days
np.array(df['Volume'].head(50))

#https://stackoverflow.com/questions/36846395/embedding-a-plotly-chart-in-a-django-template
# the graph needs formatting
def create_first_graph():
    close = [457.33401489, 424.44000244, 394.79598999, 408.9039917 ,
       398.8210144 , 402.15200806, 435.79098511, 423.20498657,
       411.57400513, 404.42498779, 399.51998901, 377.18099976,
       375.4670105 , 386.94400024, 383.61499023, 375.07199097,
       359.51199341, 328.86599731, 320.51000977, 330.07901001,
       336.18701172, 352.94000244, 365.02600098, 361.56201172,
       362.29901123, 378.54901123, 390.41400146, 400.86999512,
       394.77301025, 382.55599976, 383.75799561, 391.44198608,
       389.54598999, 382.84500122, 386.4750061 , 383.1579895 ,
       358.41699219, 358.34500122, 347.27099609, 354.70401001,
       352.98901367, 357.61801147, 335.59100342, 345.30499268,
       338.3210144 , 325.74899292, 325.89199829, 327.5539856 ,
       330.49200439, 339.48599243]
    
    date = ['2014-09-17', '2014-09-18', '2014-09-19', '2014-09-20',
       '2014-09-21', '2014-09-22', '2014-09-23', '2014-09-24',
       '2014-09-25', '2014-09-26', '2014-09-27', '2014-09-28',
       '2014-09-29', '2014-09-30', '2014-10-01', '2014-10-02',
       '2014-10-03', '2014-10-04', '2014-10-05', '2014-10-06',
       '2014-10-07', '2014-10-08', '2014-10-09', '2014-10-10',
       '2014-10-11', '2014-10-12', '2014-10-13', '2014-10-14',
       '2014-10-15', '2014-10-16', '2014-10-17', '2014-10-18',
       '2014-10-19', '2014-10-20', '2014-10-21', '2014-10-22',
       '2014-10-23', '2014-10-24', '2014-10-25', '2014-10-26',
       '2014-10-27', '2014-10-28', '2014-10-29', '2014-10-30',
       '2014-10-31', '2014-11-01', '2014-11-02', '2014-11-03',
       '2014-11-04', '2014-11-05']
    
    volume = [21056800, 34483200, 37919700, 36863600, 26580100, 24127600,
       45099500, 30627700, 26814400, 21460800, 15029300, 23613300,
       32497700, 34707300, 26229400, 21777700, 30901200, 47236500,
       83308096, 79011800, 49199900, 54736300, 83641104, 43665700,
       13345200, 17552800, 35221400, 38491500, 25267100, 26990000,
       13600700, 11416800,  5914570, 16419000, 14188900, 11641300,
       26456900, 15585700, 18127500, 11272500, 13033000,  7845880,
       18192700, 30177900, 12545400, 16677200,  8603620, 12948500,
       15655500, 19817200]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # Lineplots of price and moving averages
    fig.add_trace(go.Scatter(
                            x = date,
                            y = close,
                            mode='lines'), row = 1, col = 1)

    # Barplot of volume
    fig.add_trace(go.Bar(x = date,
                    y = volume), row = 2, col = 1)
    
    fig.show()
    
#    graph_html = fig.to_html(full_html=False, default_height=500, default_width=700)
#    ## testing the HTML
#    #print(graph_html)
#
#
#    graph_json = fig.to_json()
#    ## testing the json
#    #print(graph_json)


create_first_graph()
