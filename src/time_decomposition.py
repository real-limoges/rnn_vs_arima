'''
This modules explores the differences between traditional ARIMA modeling
and Recurrant Neural Networks
'''

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from import_data import arrange_time_series

np.random.seed(1234)

def plot_acf_pacf(ts, periods=0):
    '''
    Plots the acf and pacf
    '''
    
    fig = plt.figure( figsize=(12,8) )
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts[periods:], lags=28, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts[periods:], lags=28, ax=ax2)

    plt.show()
    plt.close()    


def extract_dow(ts):
    '''
    Extracts the day of the seek from the time series
    '''

    dow = pd.Series(pd.DatetimeIndex(ts.index).weekday)
    return dow


def create_train_test(dataset):
    '''
    Creates a train test split for time series data (univariate)
    '''

    train_size = int( len(dataset) * 0.67 )
    test_size = 1 - train_size

    return dataset[0:train_size], dataset[train_size:]


def run_ARIMA(ts, order, exog=None):
    '''
    Creates the ARIMA model
    '''

    return sm.tsa.ARIMA(ts, order=order, exog=exog).fit()
    

if __name__ == '__main__':
    logins = arrange_time_series(rule='H')
    dataset  = logins.diff(1).values.astype('float32')[1:]

    train_size = int( len(dataset) * 0.67 )
    test_size = 1 - train_size
    
    train, test = create_train_test(dataset)

    #plot_acf_pacf(dataset, 1)

    model = run_ARIMA(train, order=(0,1,0))
    print model.summary() 

    trainPredict = model.predict(1, len(train)+1, dynamic=True)
    testPredict = model.predict(1+len(train), len(train)+len(test), dynamic=True)

    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[0:len(trainPredict)] = trainPredict

    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan
    trainPredictPlot[len(trainPredict)-1:len(trainPredict)+len(testPredict)-1] = testPredict

    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
