import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from skforecast.direct import ForecasterDirect
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from lightgbm import LGBMRegressor
import os
import numpy as np
from datetime import datetime
from zephyrcast import project_config
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.utils import save_forecaster
from sklearn.feature_selection import RFECV
from skforecast.feature_selection import select_features
import ipdb

from zephyrcast.utils import load_data_from_csv


def _find_best_features(data_train, forecast_feature):
    steps = 6
    print("Find best features...")
    forecaster = ForecasterDirectMultiVariate(
        regressor=LGBMRegressor(random_state=123, verbose=-1),
        level=forecast_feature,
        steps=steps,
        lags=7,
        window_features=RollingFeatures(stats=["mean"], window_sizes=[7]),
    )

    def search_space(trial):
        search_space = {
            "lags": trial.suggest_categorical("lags", [6, 24]),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            'max_depth': trial.suggest_int("max_depth", 3, 10),
        }

        return search_space

    initial_train_data = data_train.index.max() - pd.Timedelta(days=60)
    initial_train_size = len(data_train.loc[:initial_train_data])

    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=initial_train_size,
        refit=False,
        allow_incomplete_fold=True,
    )

    results, best_trial = bayesian_search_forecaster_multiseries(
        forecaster=forecaster,
        series=data_train,
        exog=None,
        search_space=search_space,
        cv=cv,
        metric="mean_absolute_error",
        aggregate_metric="weighted_average",
        n_trials=10,
        random_state=123,
        return_best=False,
        n_jobs="auto",
        verbose=False,
        show_progress=True,
        kwargs_create_study={},
        kwargs_study_optimize={},
    )

    print(results, best_trial)
    return results, best_trial


def _get_train_test_data(
    filename: str, train_split_date: str, forecast_feature: str, save_plots=False
):
    output_dir = project_config["output_dir"]
    data = load_data_from_csv(os.path.join(output_dir, filename))

    data_train = data.loc[:train_split_date]
    data_test = data.loc[train_split_date:]

    print(
        f"Train: {data_train.index.min()} -> {data_train.index.max()} (n={len(data_train)})"
    )
    print(
        f"Test: {data_test.index.min()} -> {data_test.index.max()} (n={len(data_test)})"
    )

    if save_plots:
        graphs = 20
        fig, axes = plt.subplots(nrows=graphs, ncols=1, figsize=(9, 5), sharex=True)

        for i, col in enumerate(data.columns[:graphs]):
            print(f"Plotting: {col}")
            data_train[col].plot(ax=axes[i], label="train")
            data_test[col].plot(ax=axes[i], label="test")
            axes[i].set_ylabel("")
            axes[i].set_title(col)
            axes[i].legend(loc="upper right")

        # fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "forecast_results.png"), dpi=300)

    return data_train, data_test


def _save_model(model):
    creation_date = datetime.strptime(
        model.creation_date, "%Y-%m-%d %H:%M:%S"
    ).strftime("%Y%m%d_%H%M%S")

    ipdb.set_trace()

    model_filename = f"zephyrcast_{creation_date}_X{model.level}_S{len(model.series_names_in_)}_L{len(model.lags)}"

    models_dir = project_config["models_dir"]
    model_path = os.path.join(models_dir, f"{model_filename}.joblib")
    save_forecaster(model, file_name=model_path, verbose=True)
    print(f"Saved model: {model_path}")


def _train_model(
    data_train,
    forecast_feature: str,
    series_features: str,
    window_features,
    lags: int,
    steps: int,
):
    print(f"Forecast feature: {forecast_feature}")
    print(f"Series features: {series_features}")
    print(f"Window features: {window_features}")
    print(f"Lags: {lags}")

    # # Trial 1
    # model = ForecasterDirectMultiVariate(
    #     regressor=LGBMRegressor(
    #         n_estimators=900, random_state=15926, max_depth=7, verbose=-1
    #     ),
    #     window_features=window_features,
    #     lags=24,
    #     steps=steps,
    #     n_jobs="auto",
    #     level=forecast_feature,
    # )
    ## Trial 2
    model = ForecasterDirectMultiVariate(
        regressor=LGBMRegressor(
            n_estimators=101, random_state=15926, max_depth=6, verbose=-1
        ),
        window_features=window_features,
        lags=lags,
        steps=steps,
        n_jobs="auto",
        level=forecast_feature,
    )

    print("Fitting...")
    model.fit(series=data_train)

    feature_importance_step = 1
    top_features = 10
    print(f"Feature importance (step={feature_importance_step}, n=10):")
    print(model.get_feature_importances(step=feature_importance_step)[:top_features])

    return model


def _test_model(model, data_train, data_test, save_plots):
    print("Testing...")
    output_dir = project_config["output_dir"]
    predictions = model.predict()
    actuals = data_test.loc[predictions.index][model.level]
    error_mse = mean_squared_error(y_true=actuals, y_pred=predictions)
    print(f"Test error (MSE): {error_mse}")

    if save_plots:
        context = 500
        fig, ax = plt.subplots(figsize=(6, 3))
        data_train[-context:][model.level].plot(ax=ax, label="train")
        data_test[model.level].plot(ax=ax, label="test")
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

    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster=model,
        series=data_test,
        levels=model.level,
        cv=cv,
        metric="mean_absolute_error",
        n_jobs="auto",
        verbose=False,
        show_progress=True,
    )

    # print(metrics)
    # print(predictions)
    print("Backtesting error (MSE):", metrics)


def train():
    forecast_feature = "0_wind_avg"
    data_train, data_test = _get_train_test_data(
        filename="rocky_gully_near_6_features.csv",
        train_split_date="2025-02-07 23:59:00",
        forecast_feature=forecast_feature,
    )

    auto_parameter_tuning = False

    lags = 24
    steps = 6  # 1 hour
    window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])
    series_features = list(data_train.drop(columns=forecast_feature))
    if auto_parameter_tuning:
        results, best_trial = _find_best_features(
            data_train=data_train, forecast_feature=forecast_feature
        )
        ipdb.set_trace()
    else:

        model = _train_model(
            data_train=data_train,
            forecast_feature=forecast_feature,
            series_features=series_features,
            window_features=window_features,
            lags=lags,
            steps=steps,
        )

        _save_model(model=model)

        _test_model(
            model=model,
            data_train=data_train,
            data_test=data_test,
            save_plots=True,
        )
