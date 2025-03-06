import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skforecast.direct import ForecasterDirect
from skforecast.preprocessing import RollingFeatures
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import os
from zephyrcast import project_config


def _load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    df["t_datetime"] = pd.to_datetime(df["t_datetime"])
    df.set_index("t_datetime", inplace=True)

    df = df.asfreq("10Min")
    df = df.sort_index()

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def run():
    output_dir = project_config["output_dir"]
    data = _load_data_from_csv(os.path.join(output_dir, "rocky_gully_features.csv"))

    target = "0_temp"

    lags = 12
    steps = 120

    data_train = data[:-steps]
    data_test = data[-steps:]

    window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])

    forecaster = ForecasterDirect(
        regressor=LGBMRegressor(random_state=123, verbose=-1),
        steps=steps,
        lags=lags,
        window_features=window_features,
    )

    print("Fitting...")
    forecaster.fit(y=data_train[target])

    print("Predicting...")

    predictions = forecaster.predict()

    print(predictions.head(3))

    error_mse = mean_squared_error(y_true=data_test[target], y_pred=predictions)

    print(f"Test error (mse): {error_mse}")

    context = 500
    fig, ax = plt.subplots(figsize=(6, 3))
    data_train[-context:][target].plot(ax=ax, label="train")
    data_test[target].plot(ax=ax, label="test")
    predictions.plot(ax=ax, label="predictions")
    ax.legend()

    plt.savefig(os.path.join(output_dir, "forecast_results.png"), dpi=300)
