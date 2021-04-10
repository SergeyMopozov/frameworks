"""

# Timeseries modeling
Modeling is a process that contain creating and fitting model, that could be use for explain observation,
forecasting future values, classification etc. Here we discuss next topics:

1. Modeling approaches for 1 step forecast
2. Modeling approaches for multisteps short-term forcasting
3. Modeling approaches for multisteps long-term forcasting
4. External libraries Prophet, Orbit, Darts

"""
import numpy as np


def baseline_forecast(series, steps=1, ftype='last', period=1, season=7, n_seasons=1, recursive=False):
    """

    :param recursive:
    :param n_seasons:
    :param season:
    :param period:
    :param ftype: 'last', 'mean', 'season mean'
    :param series:
    :param steps:
    :return:
    """
    series = np.array(series)
    if ftype == 'last':
        return [series[-1] for _ in range(steps)]

    if ftype == 'mean':
        return [np.mean(series[-period:]) for _ in range(steps)]

    if ftype == 'season mean':
        forecast = []
        for step in range(steps):
            forecast.append(np.mean([series[-(season * (n + 1)) + step % season] for n in range(n_seasons)]))
        return forecast



# def optimizeSARIMA(parameters_list, d, D, s, endog, exog=None):
#     """
#         Return dataframe with parameters and corresponding AIC
#
#         parameters_list - list with (p, q, P, Q) tuples
#         d - integration order in ARIMA model
#         D - seasonal integration order
#         s - length of season
#     """
#
#     results = []
#     best_aic = float("inf")
#
#     for param in tqdm(parameters_list):
#         # we need try-except because on some combinations model fails to converge
#         try:
#             model = sm.tsa.statespace.SARIMAX(endog, exog=exog, order=(param[0], d, param[1]),
#                                               seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
#         except:
#             continue
#         aic = model.aic
#         # saving best model, AIC and parameters
#         if aic < best_aic:
#             best_model = model
#             best_aic = aic
#             best_param = param
#         results.append([param, model.aic])
#
#     result_table = pd.DataFrame(results, columns=['parameters', 'aic'])
#
#     # sorting in ascending order, the lower AIC is - the better
#     result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
#
#     return best_model, result_table