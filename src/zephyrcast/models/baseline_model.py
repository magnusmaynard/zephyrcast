from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from skforecast.model_selection import backtesting_forecaster
from skforecast.recursive import ForecasterRecursive


class NaiveRegressor:
    def fit(self, X, y):
        pass  # No fitting needed for naive model

    def predict(self, X):
        import ipdb; ipdb.set_trace()
        return X.iloc[-1]  # Predict the last known value


class BaselineModel:
    def __init__(self, steps: int, target: str):
        self._steps = steps
        self._target = target

    def _get_model(self, data) -> ForecasterRecursive:
        model = ForecasterRecursive(
            regressor=NaiveRegressor(),
            lags=1
        )
        model.fit(y=data[self._target])
        return model


    def train(self, data_train):
        pass

    def predict(self, data_test, start_date: datetime):
        model =  self._get_model(data=data_test)
        data= data_test

        freq = data.index.freq

        # TODO: assumes contiguous lags
        last_window_start = start_date - ((len(model.lags) + 1) * freq)
        last_window_end = start_date - freq
        last_window_data = data.loc[last_window_start:last_window_end]

        print(f"Last window: {last_window_start} to {last_window_end}")

        predictions = model.predict(
            last_window=last_window_data, steps=self._steps
        )[self._target]
        actuals = data.loc[predictions.index, self._target]

        self._plot_predictions_vs_actuals(
            predictions=predictions,
            actuals=actuals,
            historical_data=data.loc[
                last_window_start:last_window_end, self._target
            ],
            start_date=start_date,
            target_variable=self._target,
            save_plot=True,
        )

    # def evaluate(self, data_test: pd.DataFrame):
    #     metric, predictions = backtesting_forecaster(
    #         forecaster=self._get_model(),
    #         y=data[target_column],
    #         initial_train_size=initial_train_size,
    #         steps=steps,
    #         metric='mean_squared_error',
    #         refit=False,
    #         verbose=True
    #     )
    #     print(f"Backtesting MSE: {metric}")
    #     return metric, predictions