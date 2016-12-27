import pandas as pd

def arrange_time_series(rule='D'):
    '''
    Arranges the login time stamps into buckets given the rule. 
    Returns a pandas series.
    '''
    
    raw_logins = pd.read_json('../data/logins.json', typ='series')
    logins_adder = pd.Series(1, raw_logins) 
    
    return logins_adder.resample(rule=rule).count()

