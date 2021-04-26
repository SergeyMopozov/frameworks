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
import numpy as np


def generate_lags(timeseries, window_size=1):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with lagged feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    '''

    result = pd.DataFrame(index=timeseries.index)
    for i in range(0, window_size):
        result[f'lag_{i+1}'] = timeseries.shift(i)
    return result


def generate_diffs(timeseries, window_size=1):
    '''
    Function get univariate timeseries dataframe on input and return new dataframes with difference feature
    :type window_size: int
    :type timeseries: pandas.DataFrame
    :return: new dataframe
    '''

    result = pd.DataFrame(index=timeseries.index)
    for i in range(0, window_size):
        result[f'diff_{i+1}'] = timeseries.diff(i)
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
    :param shifts:
    :param statistics:
    :type window_sizes: int
    :type timeseries: pandas.DataFrame
    :return: new dataframe
    '''

    result = pd.DataFrame(index=timeseries.index)
    for w in window_sizes:
        for s in shifts:
            result[f'exp_{w}_{statistics}'] = timeseries.shift(s).expanding(w).agg(statistics)

    return result


def generate_trend(timeseries, kind='linear'):
    """

    :param kind:
    :param timeseries:
    :return:
    """
    if kind == 'linear':
        return np.arange(len(timeseries))
    if kind == 'quadratic':
        return np.square(np.arange(len(timeseries)))
    if kind == 'exp':
        return np.exp(np.arange(len(timeseries)))
    if kind =='log':
        return np.log1p(np.arange(len(timeseries)))
    # TODO add code for logit and Gomertz trend



def generaete_seasonality(timeseries, n, freq='H', yearly=False, monthly=False, weekly=False, dayly=False):
    """

    :param timeseries:
    :param n - Fouirie order
    :param freq:
    :param yearly:
    :param monthly:
    :param weekly:
    :param dayly:
    :return:
    """

    result = pd.DataFrame(index=timeseries.index)
    if freq == 'H':
        week_period = 7 * 24
        month_period = 30.5*24
        year_period = 365.25 * 24

    if freq == 'D':
        week_period = 7
        month_period = 30.5
        year_period = 365.25

    # features for weekly seasonality
    if weekly:
        for K in range(1, n):
            result[f'sin_week_season_{K}'] = np.sin(np.arange(len(timeseries)) * 2 * np.pi * K / week_period)
            result[f'cos_week_season_{K}'] = np.cos(np.arange(len(timeseries))* 2 * np.pi * K / week_period)

    # features for monthly seasonality
    if monthly:
        for K in range(1, n):
            result[f'sin_month_season_{K}'] = np.sin(np.arange(len(timeseries)) * 2 * np.pi * K / month_period)
            result[f'cos_month_season_{K}'] = np.cos(np.arange(len(timeseries))* 2 * np.pi * K / month_period)

    # features for yearly seasonality
    if yearly:
        for K in range(1, n):
            result[f'sin_year_season_{K}'] = np.sin(np.arange(len(timeseries)) * 2 * np.pi * K / year_period)
            result[f'cos_year_season_{K}'] = np.cos(np.arange(len(timeseries)) * 2 * np.pi * K / year_period)

    return result


def direct_forecast_feature(data, target, steps=1):
    """
    :param target:  name of target columns
    :param data: row data - unvaried time series pandas
    :param steps: number of predicted steps
    :return: features dict, targets dict
    """

    targets = {}
    features = {}
    for i in range(1, steps + 1):
        targets[i] = data[target].shift(i * (-1)).dropna()
        features[i] = data.loc[targets[i].index]

    return features, targets


def future_endog_feature(data, new_observation):
    """
    function generate endogenous features with new observation
    lags - add new observation and shift new series
    diff - same as a lags, but diff new series
    trend - add 1 for each observations and recalculate non-linear trends
    seasons - same as trends
    rolling statistics
    :param data:
    :param new_observation:
    :return:
    """

    return None


def get_future_ts(timeseries, steps=1, freq='D'):
    """
    add future to one ts or for multiple ts in pivot format (time by rows)
    :param timeseries:
    :param steps:
    :param freq:
    :return:
    """
    future_idx = pd.date_range(timeseries.index[-1], periods=steps+1, freq=freq)[1:]
    future_df = pd.DataFrame(columns=timeseries.columns, index=future_idx)
    return future_df


def get_future_mts(train_df, index_columns, target, time_index, start, end, freq='H'):
    """
    add future for multiple time series in melting format
    :param train_df:
    :param index_columns:
    :param target:
    :param time_index:
    :param start:
    :param end:
    :param freq:
    :return:
    """
    future_df = pd.DataFrame()
    for i in pd.date_range(start=start, end=end, freq=freq):
        temp_df = train_df[index_columns]
        temp_df = temp_df.drop_duplicates()

        temp_df[time_index] = i

        temp_df[target] = np.nan
        future_df = pd.concat([future_df, temp_df])

    return future_df


def code_mean(data, cat_features, real_feature):
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over real_feature

    :param data:
    :param cat_features:
    :param real_feature:
    """
    return dict(data.groupby(cat_features)[real_feature].mean())


def get_code_mean(data, target, category_list):
    """
    only for single timeseries
    :param data:
    :param target:
    :param category_list:
    :param test_size:
    :return:
    """

    for cat in category_list:
        data['average' + f'_{cat}'] = list(map(code_mean(data, cat, target).get, data[cat]))
    return data
