'''
Timeseries  forecast model quality merics

1. Simple univariate timeseries metrics (MAE, RMSE, MAPE, SMAPE)
2. Multiple timeseries metrics
3. Hiereachical timeseries metrics (Weighted Root Mean Squared Scaled Error (RMSSE))

'''
import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    '''
    Get true and predicted values and return mean absolute value
    :param y_true: array
    :param y_pred: array
    :return: int
    '''
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    '''
    Get true and predicted values and return root mean squared error
    :param y_true: array
    :param y_pred: array
    :return:
    '''
    return np.mean((y_true - y_pred) ** 2) ** 0.5


def mape(y_true, y_pred, simple=False):
    '''
    Get true and predicted values and return mean absolut percentage error,
    if true values contain zeroes use simple=True
    :param y_true:
    :param y_pred:
    :return:
    '''
    if not simple:
        return np.mean(np.abs((y_true - y_pred) / y_true))
    else:
        return np.mean(np.abs((y_true - y_pred) / np.mean(y_true)))


def resampled_error(y_true, y_pred, new_freq, error):
    '''

    :param y_true: pandas timeseries
    :param y_pred: array
    :param new_freq: new frequency
    :param error: error function
    :return:
    '''
    y_pred = pd.Series(y_pred, index=y_true.index)
    y_pred_new = y_pred.resample(new_freq).sum()
    y_true_new = y_pred.resample(new_freq).sum()

    return error(y_true_new, y_pred_new)



