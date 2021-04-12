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


def generate_trend(timeseries, linear=True, quadratic=False, exp=False, log=False,  logit=False):
    """

    :param timeseries:
    :param linear:
    :param quadratic:
    :param exp:
    :param logit:
    :param log:
    :return:
    """
    result = pd.DataFrame(index=timeseries.index)
    if linear:
        result['linear_trend'] = np.arange(len(timeseries))
    if quadratic:
        result['quadratic_trend'] = np.square(np.arange(len(timeseries)))
    if exp:
        result['quadratic_trend'] = np.exp(np.arange(len(timeseries)))
    if log:
        result['quadratic_trend'] = np.log1p(np.arange(len(timeseries)))
    # TODO add code for logit calculate
    return result


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
            result['sin_week_season_' + str(K)] = np.sin(np.arange(len(timeseries)) * 2 * np.pi * K / week_period)
            result['cos_week_season_' + str(K)] = np.cos(np.arange(len(timeseries))* 2 * np.pi * K / week_period)

    # features for monthly seasonality
    if monthly:
        for K in range(1, n):
            result['sin_month_season_' + str(K)] = np.sin(np.arange(len(timeseries)) * 2 * np.pi * K / month_period)
            result['cos_month_season_' + str(K)] = np.cos(np.arange(len(timeseries))* 2 * np.pi * K / month_period)

    # features for yearly seasonality
    if yearly:
        for K in range(1, n):
            result['sin_year_season_' + str(K)] = np.sin(np.arange(len(timeseries)) * 2 * np.pi * K / year_period)
            result['cos_year_season_' + str(K)] = np.cos(np.arange(len(timeseries)) * 2 * np.pi * K / year_period)

    return result


def direct_forecast_feature(data, target, steps=1):
    """
    :param target:  name of target columns
    :param data: row data - unvaried time series pandas
    :param steps: number of predicted steps
    :return: fitted model, forecasts
    """

    targets = {}
    features = {}
    for i in range(1, steps + 1):
        targets[i] = data[target].shift(i * (-1)).dropna()
        features[i] = data.loc[targets[i].index]

    return features, targets


