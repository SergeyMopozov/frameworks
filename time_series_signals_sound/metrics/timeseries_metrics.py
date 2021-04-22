"""
Timeseries  forecast model quality merics

1. Simple univariate timeseries metrics (MAE, RMSE, MAPE, SMAPE)
2. Multiple timeseries metrics
3. Hiereachical timeseries metrics (Weighted Root Mean Squared Scaled Error (RMSSE))

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def mae(y_true, y_pred):
    """
    Get true and predicted values and return mean absolute value
    :param y_true: array
    :param y_pred: array
    :return: int
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    Get true and predicted values and return root mean squared error
    :param y_true: array
    :param y_pred: array
    :return:
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2) ** 0.5


def mape(y_true, y_pred, simple=False):
    """
    Get true and predicted values and return mean absolut percentage error,
    if true values contain zeroes use simple=True
    :param y_true:
    :param y_pred:
    :return:
    """
    # TODO check correctness
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if not simple:
        return np.mean(np.abs((y_true - y_pred) / y_true))
    else:
        return np.mean(np.abs((y_true - y_pred) / np.mean(y_true)))


def smape(y_true, y_pred, simple=False):
    """

    :param y_true:
    :param y_pred:
    :param simple
    :return:
    """
    # TODO check correctness
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if not simple:
        return np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred) / 2)))
    else:
        return np.mean(np.abs((y_true - y_pred) / (np.mean(y_true + y_pred) / 2)))


def weighted_error(y_true, y_pred, weight, error):
    """

    :param y_true:
    :param y_pred:
    :param weight:
    :param error:
    :return:
    """
    # TODO write function that calculate wheigted error
    return None


def resampled_error(y_true, y_pred, new_freq, error):
    """

    :param y_true: pandas timeseries
    :param y_pred: array
    :param new_freq: new frequency
    :param error: error function
    :return:
    """
    # TODO  add check on pandas sereis or create a series with datetime index
    y_pred = pd.Series(y_pred, index=y_true.index)
    y_pred_new = y_pred.resample(new_freq).sum()
    y_true_new = y_true.resample(new_freq).sum()

    return error(y_true_new, y_pred_new)


def timeseriesCVscore(series, model, loss_function, n_splits=3):
    """

    :param n_splits:
    :param series:
    :param model:
    :param loss_function:
    :return:
    """
    # errors array
    errors = []
    values = series.values

    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # iterating over folds, train model on each, forecast and calculate error
    # TODO add interface for learning
    for train, test in tscv.split(values):
        model.fit(values[train])
        predictions = model.predict(steps=len(test))
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)

    return np.mean(np.array(errors))
