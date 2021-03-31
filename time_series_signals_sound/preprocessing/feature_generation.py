'''
This source code content function for work with time series future generation routine
'''


def generate_lags(timeseries, window_size, terget_name=None):
    '''
    Function get timeseries dataframe on input and return new dataframes with lagged feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = timeseries.copy()
    for i in range(1, window_size + 1):
        result[f'lag_{i}'] = result.shift(i)
    return result


def generate_diffs(timeseries, window_size, terget_name=None):
    '''
    Function get timeseries dataframe on input and return new dataframes with difference feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = timeseries.copy()
    for i in range(1, window_size + 1):
        result[f'lag_{i}'] = result.diff(i)
    return result

def generate_rolling_statistics(timeseries, window_size, statistics, terget_name=None):
    '''
    Function get timeseries dataframe on input and return new dataframes with rolling statistic feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = timeseries.copy()
    for i in range(1, window_size + 1):
        # TODO apply different finction for rolling window df
        #result[f'lag_{i}'] = result.rolling(window_size).statistics
        continue
    return result