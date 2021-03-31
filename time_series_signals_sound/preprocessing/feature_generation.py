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


def generate_lags(timeseries, window_size, terget_name=None):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with lagged feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = timeseries.copy()
    for i in range(1, window_size + 1):
        result[f'lag_{i}'] = result.shift(i)
    return result


def generate_diffs(timeseries, window_size, terget_name=None):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with difference feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = timeseries.copy()
    for i in range(1, window_size + 1):
        result[f'lag_{i}'] = result.diff(i)
    return result

def generate_rolling_statistics(timeseries, window_size, statistics, terget_name=None):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with rolling statistic feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = timeseries.copy()
    for i in range(1, window_size + 1):
        # TODO apply different finction for rolling window df
        #result[f'lag_{i}'] = result.rolling(window_size).statistics
        continue
    return result