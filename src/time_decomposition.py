'''
This modules explores the differences between traditional ARIMA modeling
and Recurrant Neural Networks
'''

import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

def arrange_time_series(rule='D'):
    '''
    Arranges the login time stamps into buckets given the rule. 
    Returns a pandas series.
    '''
    
    raw_logins = pd.read_json('../data/logins.json', typ='series')
    logins_adder = pd.Series(1, raw_logins) 
    
    return logins_adder.resample(rule=rule).count()


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

    dow = pd.Series(pd.DatetimeIndex(ts.index).weekday)
    return dow


def run_ARIMA(ts, exog, order):
    '''

    '''

    model = sm.tsa.ARIMA(ts, order=order, exog=exog).fit()
    print model.summary()


if __name__ == '__main__':
    logins = arrange_time_series(rule='H')
    logins_diff_7 = logins.diff(1)

    dow = extract_dow(logins)

#    plot_acf_pacf(logins_diff_7, 1)


    run_ARIMA(list(logins_diff_7)[1:], list(dow[1:]), (0,1,0))

