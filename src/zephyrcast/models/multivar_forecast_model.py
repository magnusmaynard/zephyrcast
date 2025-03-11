import pandas as pd
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from lightgbm import LGBMRegressor
import os
from datetime import datetime
from zephyrcast import project_config
from skforecast.preprocessing import RollingFeatures
from skforecast.utils import save_forecaster
import ipdb
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
        self._context = 24
        self._model = self._load_latest_model()

    @property
    def context(self):
        return self._context

    def _save_model(self):
        creation_date = datetime.strptime(
            self._model.creation_date, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y%m%d_%H%M%S")

        models_dir = project_config["models_dir"]
        model_filename = f"zephyrcast_{creation_date}_X{self._model.level}_S{len(self._model.series_names_in_)}_L{len(self._model.lags)}"

        model_path = os.path.join(models_dir, f"{model_filename}.joblib")
        save_forecaster(self._model, file_name=model_path, verbose=True)
        print(f"Saved model: {model_path}")

    def _load_latest_model(self) -> ForecasterDirectMultiVariate:
        latest_model_path = find_latest_model_path()
        return load_forecaster(latest_model_path, verbose=True)

    def train(self, data_train: pd.DataFrame):
        data, data_const = extract_constant_features(data_train)
        data_exog = pd.concat([data_const] * len(data), ignore_index=True)
        data_exog.index = data.index

        window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])
        series_features = list(data.drop(columns=self._target))

        print(f"Forecast feature: {self._target}")
        print(f"Series features: {series_features}")
        print(f"Window features: {window_features}")
        print(f"Lags: {self._context}")
        print(f"Exogenous features: {list(data_exog.columns)}")

        model = ForecasterDirectMultiVariate(
            regressor=LGBMRegressor(
                n_estimators=101, random_state=15926, max_depth=6, verbose=-1
            ),
            window_features=window_features,
            lags=self._context,
            steps=self._steps,
            n_jobs="auto",
            level=self._target,
        )

        print("Fitting...")
        model.fit(series=data, exog=data_exog)

        feature_importance_step = 1
        top_features = 10
        print(f"Feature importance (step={feature_importance_step}, n=10):")
        print(
            model.get_feature_importances(step=feature_importance_step)[:top_features]
        )

        self._save_model()

    def predict(self, last_window: pd.DataFrame) -> pd.DataFrame:
        data = last_window.drop(columns=self._model.exog_names_in_)
        data_exog = last_window.drop(columns=self._model.series_names_in_)
        start_index = last_window.index[-1] + last_window.index.freq
        new_index = pd.date_range(start=start_index, periods=len(data_exog), freq=last_window.index.freq)
        data_exog.index = new_index

        return  self._model.predict(
            last_window=data, exog=data_exog
        )[self._model.level]
