import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skforecast.direct import ForecasterDirect
from skforecast.preprocessing import RollingFeatures
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import os
from datetime import datetime
from zephyrcast import project_config
import ipdb

from skforecast.datasets import fetch_dataset
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from skforecast.feature_selection import select_features
from skforecast.feature_selection import select_features_multiseries
import json


def _load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    df["t_datetime"] = pd.to_datetime(df["t_datetime"])
    df.set_index("t_datetime", inplace=True)

    df = df.asfreq("10Min")
    df = df.sort_index()

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def _find_best_features(data_train, forecast_feature, exog_features):
    print("Find best features...")
    window_features = RollingFeatures(
        stats=["mean", "mean", "sum"], window_sizes=[24, 48, 24]
    )

    forecaster = ForecasterRecursive(
        regressor=LGBMRegressor(
            n_estimators=900, random_state=15926, max_depth=7, verbose=-1
        ),
        lags=48,
        window_features=window_features,
    )

    regressor = LGBMRegressor(
        n_estimators=100, max_depth=5, random_state=15926, verbose=-1
    )

    selector = RFECV(
        estimator=regressor, step=1, cv=3, min_features_to_select=25, n_jobs=-1
    )

    selected_lags, selected_window_features, selected_exog_features = select_features(
        forecaster=forecaster,
        selector=selector,
        y=data_train[forecast_feature],
        exog=data_train[exog_features],
        select_only=None,
        force_inclusion=None,
        subsample=0.5,
        random_state=123,
        verbose=True,
    )

    return selected_lags, selected_window_features, selected_exog_features


def _get_train_test_data(filename: str, train_split_date: str, forecast_feature: str):
    output_dir = project_config["output_dir"]
    data = _load_data_from_csv(os.path.join(output_dir, filename))

    data_train = data.loc[:train_split_date]
    data_test = data.loc[train_split_date:]

    print(
        f"Train: {data_train.index.min()} -> {data_train.index.max()} (n={len(data_train)})"
    )
    print(
        f"Test: {data_test.index.min()} -> {data_test.index.max()} (n={len(data_test)})"
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    data_train[forecast_feature].plot(ax=ax, label="train")
    data_test[forecast_feature].plot(ax=ax, label="test")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "data.png"), dpi=300)

    return data_train, data_test


def _save_model(model, forecast_feature):
    creation_date = datetime.strptime(
        model.creation_date, "%Y-%m-%d %H:%M:%S"
    ).strftime("%Y%m%d_%H%M%S")

    model_filename = f"zephyrcast_{creation_date}_x{forecast_feature}_exo{len(model.exog_names_in_)}_l{model.lags}"

    models_dir = project_config["models_dir"]
    model_path = os.path.join(models_dir, f"{model_filename}.joblib")
    save_forecaster(model, file_name=model_path, verbose=True)
    print(f"Saved model: {model_path}")


def _train_model(
    data_train,
    forecast_feature: str,
    exog_features: str,
    window_features,
    lags: int,
    steps: int,
):
    print(f"Forecast feature: {forecast_feature}")
    print(f"Exo features: {exog_features}")
    print(f"Window features: {window_features}")
    print(f"Lags: {lags}")

    model = ForecasterDirect(
        regressor=LGBMRegressor(
            n_estimators=900, random_state=15926, max_depth=7, verbose=-1
        ),
        window_features=window_features,
        lags=lags,
        steps=steps,
        n_jobs="auto",
    )

    print("Fitting...")
    model.fit(y=data_train[forecast_feature], exog=data_train[exog_features])

    feature_importance_step = 1
    top_features = 10
    print(f"Feature importance (step={feature_importance_step}, n=10):")
    print(model.get_feature_importances(step=feature_importance_step)[:top_features])

    return model


def _test_model(
    model, data_train, data_test, forecast_feature, exog_features, save_plots
):
    print("Testing...")
    output_dir = project_config["output_dir"]
    predictions = model.predict(exog=data_test[exog_features])
    actuals = data_test.loc[predictions.index][forecast_feature]
    error_mse = mean_squared_error(y_true=actuals, y_pred=predictions)
    print(f"Test error (MSE): {error_mse}")

    if save_plots:
        context = 500
        fig, ax = plt.subplots(figsize=(6, 3))
        data_train[-context:][forecast_feature].plot(ax=ax, label="train")
        data_test[forecast_feature].plot(ax=ax, label="test")
        predictions.plot(ax=ax, label="predictions")
        ax.legend()

        plt.savefig(os.path.join(output_dir, "forecast_results.png"), dpi=300)

    print("Backtesting...")

    last_date = data_test.index.max()
    backtest_date = last_date - pd.Timedelta(days=10)
    initial_train_size = len(data_test.loc[:backtest_date])
    cv = TimeSeriesFold(
        steps=model.steps,
        initial_train_size=initial_train_size,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
    )

    metric, predictions = backtesting_forecaster(
        forecaster=model,
        y=data_test[forecast_feature],
        exog=data_test[exog_features],
        cv=cv,
        metric="mean_squared_error",
        n_jobs="auto",
        verbose=False,
        show_progress=True,
    )

    print("Backtesting error (MSE):", metric["mean_squared_error"][0])


def run():
    forecast_feature = "0_temp"
    data_train, data_test = _get_train_test_data(
        filename="rocky_gully_features.csv",
        train_split_date="2025-02-07 23:59:00",
        forecast_feature=forecast_feature,
    )

    auto_feature_selection = False

    lags = 24
    window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])
    exog_features = list(data_train.drop(columns=forecast_feature))
    if auto_feature_selection:
        lags, window_features, exog_features = _find_best_features(
            data_train=data_train,
            forecast_feature=forecast_feature,
            exog_features=exog_features,
        )

    model = _train_model(
        data_train=data_train,
        forecast_feature=forecast_feature,
        exog_features=exog_features,
        window_features=window_features,
        lags=lags,
        steps=6,
    )

    _save_model(model=model, forecast_feature=forecast_feature)

    _test_model(
        model=model,
        data_train=data_train,
        data_test=data_test,
        forecast_feature=forecast_feature,
        exog_features=exog_features,
        save_plots=True,
    )
