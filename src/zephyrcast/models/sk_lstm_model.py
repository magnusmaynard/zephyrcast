
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import pandas as pd
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from lightgbm import LGBMRegressor
from datetime import datetime
from zephyrcast import project_config
from skforecast.preprocessing import RollingFeatures
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
import skforecast
from skforecast.plot import set_dark_theme
from skforecast.datasets import fetch_dataset
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.model_selection import TimeSeriesFold

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping

from zephyrcast.data.utils import (
    extract_constant_features,
    find_latest_model_path,
)
from zephyrcast.models.model_interface import ModelInterface



import warnings
warnings.filterwarnings('once')

print(f"skforecast version: {skforecast.__version__}")
print(f"keras version: {keras.__version__}")

if keras.__version__ > "3.0":
    print(f"Using backend: {keras.backend.backend()}")
    if keras.backend.backend() == "tensorflow":
        import tensorflow
        print(f"tensorflow version: {tensorflow.__version__}")
    elif keras.backend.backend() == "torch":
        import torch
        print(f"torch version: {torch.__version__}")
    else:
        print("Backend not recognized. Please use 'tensorflow' or 'torch'.")


class SKLSTMModel(ModelInterface):
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
        return self._model.is_fitted

    @property
    def name(self):
        creation_date = datetime.strptime(
            self._model.creation_date, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y%m%d_%H%M%S")
        return f"sklstm_{creation_date}_X{self._target}"

    def _save_model(self):
        models_dir = project_config["models_dir"]
        model_path = os.path.join(models_dir, f"{self.name}.joblib")
        save_forecaster(self._model, file_name=model_path, verbose=True)
        print(f"Saved model: {model_path}")

    def _load_latest_model(self) -> ForecasterRnn | None:
        try:
            latest_model_path = find_latest_model_path(prefix="sklstm_", suffix=f"{self._target}.joblib")
        except FileNotFoundError:
            return None
        return load_forecaster(latest_model_path, verbose=True)

    def train(self, data_train: pd.DataFrame, data_test: pd.DataFrame):
        # data, data_const = extract_constant_features(data_train)
        # data_exog = pd.concat([data_const] * len(data), ignore_index=True)
        # data_exog.index = data.index

        # data_test.drop(columns=data_exog.columns)

        data = data_train

        # window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])
        # series_features = list(data.drop(columns=self._target))

        print(f"Forecast feature: {self._target}")
        # print(f"Series features: {series_features}")
        # print(f"Window features: {window_features}")
        print(f"Lags: {self._window_size}")
        # print(f"Exogenous features: {list(data_exog.columns)}")

        regressor = create_and_compile_model(
            series=data,
            levels=[self._target], 
            lags=self._window_size,
            steps=self._steps,
            recurrent_layer="LSTM",
            recurrent_units=50,
            dense_units=32,
            optimizer=Adam(learning_rate=0.01), 
            loss=MeanSquaredError()
        )
        regressor.summary()

        self._model = ForecasterRnn(
            regressor=regressor,
            levels=[self._target],
            transformer_series=MinMaxScaler(),
            fit_kwargs={
                "epochs": 2,
                "batch_size": 32,
                "callbacks": [
                    EarlyStopping(monitor="val_loss", patience=5)
                ],
                "series_val": data_test,
            },
        )    

        self._model 

        print("Fitting...")
        self._model.fit(series=data)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        self._model.plot_history(ax=ax)

        # feature_importance_step = 1
        # top_features = 10
        # print(f"Feature importance (step={feature_importance_step}, n=10):")
        # print(
        #     self._model.get_feature_importances(step=feature_importance_step)[:top_features]
        # )

        self._save_model()

    def predict(self, last_window: pd.DataFrame) -> pd.DataFrame:
        # data = last_window.drop(columns=self._model.exog_names_in_)
        # data_exog = last_window.drop(columns=self._model.series_names_in_)
        # start_index = last_window.index[-1] + last_window.index.freq
        # new_index = pd.date_range(
        #     start=start_index, periods=len(data_exog), freq=last_window.index.freq
        # )
        # data_exog.index = new_index

        # return self._model.predict(last_window=data, exog=data_exog)[self._model.level]
        predictions = self._model.predict(last_window=last_window)

        return predictions.drop(columns="level").squeeze()
