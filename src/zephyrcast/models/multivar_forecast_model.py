import pandas as pd
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from lightgbm import LGBMRegressor
import os
from datetime import datetime
from zephyrcast import project_config
from skforecast.preprocessing import RollingFeatures
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

from zephyrcast.data.utils import (
    extract_constant_features,
    find_latest_model_path,
)
from zephyrcast.models.model_interface import ModelInterface


class MultiVariantForecastModel(ModelInterface):
    def __init__(self, steps: int, target: str):
        self._steps = steps
        self._target = target
        self._window_size = 24
        self._model = self._load_latest_model()

    @property
    def window_size(self):
        return self._window_size

    @property
    def is_trained(self):
        return self._model is not None and self._model.is_fitted

    @property
    def name(self):
        creation_date = datetime.strptime(
            self._model.creation_date, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y%m%d_%H%M%S")
        return f"multivar_{creation_date}_X{self._target}"

    def _save_model(self):
        models_dir = project_config["models_dir"]
        model_path = os.path.join(models_dir, f"{self.name}.joblib")
        save_forecaster(self._model, file_name=model_path, verbose=True)
        print(f"Saved model: {model_path}")

    def _load_latest_model(self) -> ForecasterDirectMultiVariate | None:
        try:
            latest_model_path = find_latest_model_path(prefix="multivar", suffix=f"{self._target}.joblib")
        except FileNotFoundError:
            return None
        return load_forecaster(latest_model_path, verbose=True)

    def train(self, data_train: pd.DataFrame, data_test: None):
        data, data_const = extract_constant_features(data_train)
        data_exog = pd.concat([data_const] * len(data), ignore_index=True)
        data_exog.index = data.index

        window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])
        series_features = list(data.drop(columns=self._target))

        print(f"Forecast feature: {self._target}")
        print(f"Series features: {series_features}")
        print(f"Window features: {window_features}")
        print(f"Lags: {self._window_size}")
        print(f"Exogenous features: {list(data_exog.columns)}")

        self._model = ForecasterDirectMultiVariate(
            regressor=LGBMRegressor(
                n_estimators=101, random_state=15926, max_depth=6, verbose=-1
            ),
            window_features=window_features,
            lags=self._window_size,
            steps=self._steps,
            n_jobs="auto",
            level=self._target,
        )

        print("Fitting...")
        self._model.fit(series=data, exog=data_exog)

        feature_importance_step = 1
        top_features = 10
        print(f"Feature importance (step={feature_importance_step}, n=10):")
        print(
            self._model.get_feature_importances(step=feature_importance_step)[:top_features]
        )

        self._save_model()

    def predict(self, last_window: pd.DataFrame) -> pd.DataFrame:
        data = last_window.drop(columns=self._model.exog_names_in_)
        data_exog = last_window.drop(columns=self._model.series_names_in_)
        start_index = last_window.index[-1] + last_window.index.freq
        new_index = pd.date_range(
            start=start_index, periods=len(data_exog), freq=last_window.index.freq
        )
        data_exog.index = new_index

        predictions = self._model.predict(last_window=data, exog=data_exog)

        return predictions.drop(columns="level").squeeze()
