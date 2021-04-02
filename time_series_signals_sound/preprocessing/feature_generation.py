'''
This source code content function for work with time series preprocessing and feature generation routine

Multitimeseries data is an independent timeseries that represent similar processes or one process
 from different view points (different sensors, places etc.)

    1. Normalize data table structure
    2. Time series tasks:
        2.1 Forecasting / prediction
        2.2 Segmentation / marking / change detection
        2.3 Classification
    3. Feature generation
    4. Feature extraction
    5. External libraries for feature generation
'''

import pandas as pd

def generate_lags(timeseries, window_size=1):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with lagged feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = pd.DataFrametime(index=timeseries.index)
    for i in range(1, window_size + 1):
        result[f'lag_{i}'] = timeseries.shift(i)
    return result


def generate_diffs(timeseries, window_size=1):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with difference feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    :return: new dataframe
    '''

    result = pd.DataFrametime(index=timeseries.index)
    for i in range(1, window_size + 1):
        result[f'diff_{i}'] = timeseries.diff(i)
    return result

def generate_rolwin_stat(timeseries, window_sizes, shifts=[1], statistics='mean'):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with rolling statistic feature
    :type window_sizes: list
    :type timeseries: pandas.DataFrame
    :return: new dataframe
    '''

    result = pd.DataFrametime(index=timeseries.index)
    for w in window_sizes:
        for s in shifts:
            result[f'roll_{w}_{statistics}'] = timeseries.shift(s).rolling(w).agg(statistics)

    return result

def generate_expwin_stat(timeseries, window_sizes, shifts=[1], statistics='mean'):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with rolling statistic feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    :return: new dataframe
    '''

    result = pd.DataFrametime(index=timeseries.index)
    for w in window_sizes:
        for s in shifts:
            result[f'exp_{w}_{statistics}'] = timeseries.shift(s).expanding(w).agg(statistics)

    return result

