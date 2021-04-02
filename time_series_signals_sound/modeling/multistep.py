'''
Multi-step prediction approaches
Direct Multi-step Forecast Strategy The direct method involves developing a separate model for each forecast time step.

Example:
prediction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model2(obs(t-2), obs(t-3), ..., obs(t-n))

Recursive Multi-step Forecast The recursive strategy involves using a one-step model multiple times where
the prediction for the prior time step is used as an input for making a prediction on the following time step.

Example:
prediction(t+1) = model(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model(prediction(t+1), obs(t-1), ..., obs(t-n))

Direct-Recursive Hybrid Strategies The direct and recursive strategies can be combined to offer
the benefits of both methods.
For example, a separate model can be constructed for each time step to be predicted, but each model may use
the predictions made by models at prior time steps as input values.

Example:
prediction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model2(prediction(t+1), obs(t-1), ..., obs(t-n))

Multiple Output Strategy The multiple output strategy involves developing one model that is capable of predicting
the entire forecast sequence in a one-shot manner.
Example:

prediction(t+1), prediction(t+2) = model(obs(t-1), obs(t-2), ..., obs(t-n))
'''

from ..preprocessing import feature_generation


# direct strategy
def direct_forecast(model, data, predicted_steps=1):
    '''
    :param model: class of the ML model
    :param data: row data - univariate time series
    :param predicted_steps: number of predicted steps
    :return: fitted model, forecasts
    '''

    # 1 prepare data
    targets = {}
    features = {}
    for i in range(1, predicted_steps + 1):
        targets[i] = data.shift(i * (-1))
        features[i] = feature_generation.generate_lags(data.shift(i-1))

    # 2 fit model
    models = {}
    for i in range(1, predicted_steps + 1):
        # TODO how to implement different models
        m = model
        m.fit(features[i], targets[i])
        models[i] = m

    # 3 forecast
    forecast = {}
    # all models predict on last observation
    future_features = feature_generation.generate_lags(data).iloc[-1:]
    for key in models.keys():
        forecast[key] = models[key].predict(future_features)

    return models, forecast


# recursive strategy
def recursive_forecast(model, data, predicted_steps=1):
    '''
    :param model: class of the ML model
    :param data: row data - univariate time series
    :param predicted_steps: number of predicted steps
    :return: fitted model, forecasts
    '''

    #1 prepare data
    targets = data.shift(-1)
    features = feature_generation.generate_lags(data)

    #2 fit model
    model.fit(features, targets)

    #3 forecast
    forecast = {}
    #all models predict on last observation

    for i in range(predicted_steps):

        future_features = feature_generation.generate_lags(data).iloc[-1:]
        forecast[i] = model.predict(future_features)
        data = data.append(forecast[i])

    return model, forecast
