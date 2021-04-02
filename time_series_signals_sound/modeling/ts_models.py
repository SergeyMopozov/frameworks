'''

# Timeseries modeling
Modeling is a process that contain creating and fitting model, that could be use for explain observation, forecasting future values, classification etc. Here we discuss next topics:

1. Modeling aproaches for 1 step forecast
2. Modeling aproaches for multisteps short-term forcasting
3. Modeling aproaches for multisteps long-term forcasting
4. External libraries Prophet, Orbit, Darts

'''


def optimizeSARIMA(parameters_list, d, D, s, endog, exog):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    # results = []
    # best_aic = float("inf")
    #
    # for param in tqdm(parameters_list):
    #     # we need try-except because on some combinations model fails to converge
    #     try:
    #         model = sm.tsa.statespace.SARIMAX(endog, exog=exog, order=(param[0], d, param[1]),
    #                                           seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
    #     except:
    #         continue
    #     aic = model.aic
    #     # saving best model, AIC and parameters
    #     if aic < best_aic:
    #         best_model = model
    #         best_aic = aic
    #         best_param = param
    #     results.append([param, model.aic])
    #
    # result_table = pd.DataFrame(results, columns=['parameters', 'aic'])
    #
    # # sorting in ascending order, the lower AIC is - the better
    # result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    #
    # return result_table