"""
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
"""

from ..preprocessing import feature_generation



def direct_forecast_feature(data, predicted_steps=1):
    """
    :param model: class of the ML model
    :param data: row data - unvaried time series
    :param predicted_steps: number of predicted steps
    :return: fitted model, forecasts
    """

    # 1 prepare data
    targets = {}
    features = {}
    for i in range(1, predicted_steps + 1):
        targets[i] = data.shift(i * (-1))
        features[i] = feature_generation.generate_lags(data.shift(i-1))

    return features, targets


# direct strategy
class DirectForecast:
    def __init__(self, model, steps=1):

        self.models = []
        self.steps = steps
        for i in self.steps:
            self.models.append(model)

    def fit(self, X_train, y_train):
        for m in self.models:
            m.fit(X_train, y_train)

    def predict(self, X_future):
        forecast = []
        for m in self.models:
            forecast.append(m.predict(X_future))

        return forecast


# recursive strategy
class RecursiveForecast:
    def __init__(self, model, feature_generator, steps=1):
        self.model = model
        self.feature_generator = feature_generator
        self.steps = steps

    def fit(self, series):
        X_train = self.feature_generator(series)
        y_train = series
        self.model.fit(X_train, y_train)

    def predict(self, series):
        #TODO  for predictin need recalculate features frame for next prediction
        # think about how optimize this process
        forecast = []
        for i in range(self.steps):
            if i == 0:
                features = self.feature_generator(series)
                forecast.append(self.model.predict(features[-1]))
            else:
                # 1 add previous forecasted value
                series.append(forecast[i - 1])
                # 2 recalculate features
                features = self.feature_generator(series)
                # 3 predict
                forecast.append(self.model.predict(features[-1]))






