from click import UsageError
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from lightgbm import LGBMRegressor
import os
import numpy as np
from datetime import datetime
from zephyrcast import project_config
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.utils import save_forecaster
import ipdb
from skforecast.utils import load_forecaster

from zephyrcast.utils import (
    extract_constant_features,
    find_latest_model_path,
    load_data_from_csv,
)


class MultiVariantForecastModel:
    def __init__(self):
        self._model = self._load_latest_model()

    def _find_best_features(self, data_train, exog, forecast_feature):
        steps = 6
        print("Find best features...")
        forecaster = ForecasterDirectMultiVariate(
            regressor=LGBMRegressor(random_state=123, verbose=-1),
            level=forecast_feature,
            steps=steps,
            lags=7,
            window_features=RollingFeatures(stats=["mean"], window_sizes=[7]),
        )

        def search_space(self, trial):
            search_space = {
                "lags": trial.suggest_categorical("lags", [6, 24]),
                "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
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
            exog=exog,
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
        self,
        filename: str,
        train_split_date: str,
        forecast_feature: str,
        save_plots=False,
    ):
        output_dir = project_config["output_dir"]
        data = load_data_from_csv(os.path.join(output_dir, filename))

        data, constant_features = extract_constant_features(data)

        data_train = data.loc[:train_split_date]
        data_test = data.loc[train_split_date:]

        print(
            f"Train: {data_train.index.min()} -> {data_train.index.max()} (n={len(data_train)})"
        )
        print(
            f"Test: {data_test.index.min()} -> {data_test.index.max()} (n={len(data_test)})"
        )

        if save_plots:
            graphs = min(20, len(data.columns))
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

        # Create exog for train and test sets with the same constant features
        exog_train = pd.concat([constant_features] * len(data_train), ignore_index=True)
        exog_train.index = data_train.index

        exog_test = pd.concat([constant_features] * len(data_test), ignore_index=True)
        exog_test.index = data_test.index

        return data_train, data_test, exog_train, exog_test, constant_features

    def _save_model(self, constant_features):
        creation_date = datetime.strptime(
            self._model.creation_date, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y%m%d_%H%M%S")

        # Save constant features alongside the model
        models_dir = project_config["models_dir"]
        constants_filename = f"zephyrcast_{creation_date}_constants.csv"
        constant_features.to_csv(
            os.path.join(models_dir, constants_filename), index=False
        )
        print(f"Saved constant features: {constants_filename}")

        ipdb.set_trace()

        model_filename = f"zephyrcast_{creation_date}_X{self._model.level}_S{len(self._model.series_names_in_)}_L{len(self._model.lags)}"

        model_path = os.path.join(models_dir, f"{model_filename}.joblib")
        save_forecaster(self._model, file_name=model_path, verbose=True)
        print(f"Saved model: {model_path}")

    def _train_model(
        self,
        data_train,
        exog_train,
        forecast_feature: str,
        series_features: str,
        window_features,
        lags: int,
        steps: int,
    ) -> ForecasterDirectMultiVariate:
        print(f"Forecast feature: {forecast_feature}")
        print(f"Series features: {series_features}")
        print(f"Window features: {window_features}")
        print(f"Lags: {lags}")
        print(f"Exogenous features: {list(exog_train.columns)}")

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
        model.fit(series=data_train, exog=exog_train)

        feature_importance_step = 1
        top_features = 10
        print(f"Feature importance (step={feature_importance_step}, n=10):")
        print(
            model.get_feature_importances(step=feature_importance_step)[:top_features]
        )

        return model

    def _test_model(self, data_train, data_test, exog_test, save_plots):
        print("Testing...")
        output_dir = project_config["output_dir"]
        predictions = self._model.predict(exog=exog_test)
        actuals = data_test.loc[predictions.index][self._model.level]
        error_mse = mean_squared_error(y_true=actuals, y_pred=predictions)
        print(f"Test error (MSE): {error_mse}")

        if save_plots:
            context = 500
            fig, ax = plt.subplots(figsize=(6, 3))
            data_train[-context:][self._model.level].plot(ax=ax, label="train")
            data_test[self._model.level].plot(ax=ax, label="test")
            predictions.plot(ax=ax, label="predictions")
            ax.legend()

            plt.savefig(os.path.join(output_dir, "forecast_results.png"), dpi=300)

        print("Backtesting...")

        last_date = data_test.index.max()
        backtest_date = last_date - pd.Timedelta(days=10)
        initial_train_size = len(data_test.loc[:backtest_date])
        cv = TimeSeriesFold(
            steps=self._model.steps,
            initial_train_size=initial_train_size,
            refit=False,
            fixed_train_size=False,
            gap=0,
            allow_incomplete_fold=True,
        )

        metrics, predictions = backtesting_forecaster_multiseries(
            forecaster=self._model,
            series=data_test,
            exog=exog_test,
            levels=self._model.level,
            cv=cv,
            metric="mean_absolute_error",
            n_jobs="auto",
            verbose=False,
            show_progress=True,
        )

        print("Backtesting error (MSE):", metrics)

    def _load_latest_model(self) -> ForecasterDirectMultiVariate:
        latest_model_path = find_latest_model_path()
        return load_forecaster(latest_model_path, verbose=True)

    def _check_dates(self, start):
        training_start = self._model.training_range_[0]
        training_end = self._model.training_range_[1]

        if training_start <= start <= training_end:
            raise UsageError(
                f"The testing date {start.strftime('%Y-%m-%d %H:%M:%S')} is within the training range {training_start} -> {training_end}. Please choose a date outside this range."
            )

    def _plot_predictions_vs_actuals(
        self,
        predictions,
        actuals,
        historical_data,
        start_date,
        target_variable,
        save_plot=False,
    ):
        # Calculate error metrics
        comparison_df = pd.DataFrame({"Predicted": predictions, "Actual": actuals})
        mae = (comparison_df["Predicted"] - comparison_df["Actual"]).abs().mean()
        rmse = (
            (comparison_df["Predicted"] - comparison_df["Actual"]) ** 2
        ).mean() ** 0.5

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(
            historical_data.index,
            historical_data.values,
            color="blue",
            label="Historical Data",
        )

        # Plot actual values
        plt.plot(
            actuals.index,
            actuals.values,
            color="green",
            linewidth=2,
            label="Actual Values",
        )

        # Plot predictions
        plt.plot(
            predictions.index,
            predictions.values,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Predicted Values",
        )

        # Add vertical line to mark the start of predictions
        plt.axvline(x=start_date, color="black", linestyle="-", alpha=0.7)
        plt.text(
            start_date,
            plt.ylim()[1] * 0.9,
            "Prediction Start",
            rotation=90,
            verticalalignment="top",
        )

        # Customize the plot
        plt.title(
            f"Forecast vs Actual Values for {target_variable}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}"
        )
        plt.xlabel("Date")
        plt.ylabel(target_variable)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Adjust x-axis date formatting
        plt.gcf().autofmt_xdate()

        # Save the plot if requested
        if save_plot:
            plot_dir = os.path.join(project_config.get("output_dir", ""), "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(
                plot_dir,
                f'forecast_comparison_{start_date.strftime("%Y%m%d_%H%M%S")}.png',
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {plot_path}")

        plt.tight_layout()
        plt.show()

    def train(self):
        forecast_feature = "0_wind_avg"
        data_train, data_test, exog_train, exog_test, constant_features = (
            self._get_train_test_data(
                filename="rocky_gully_near_6_features.csv",
                train_split_date="2025-02-07 23:59:00",
                forecast_feature=forecast_feature,
            )
        )

        auto_parameter_tuning = False

        lags = 24
        steps = 6  # 1 hour
        window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])
        series_features = list(data_train.drop(columns=forecast_feature))
        if auto_parameter_tuning:
            results, best_trial = self._find_best_features(
                data_train=data_train,
                exog=exog_train,
                forecast_feature=forecast_feature,
            )
            ipdb.set_trace()
        else:
            self._model = self._train_model(
                data_train=data_train,
                exog_train=exog_train,
                forecast_feature=forecast_feature,
                series_features=series_features,
                window_features=window_features,
                lags=lags,
                steps=steps,
            )

            self._save_model(constant_features=constant_features)

            self._test_model(
                data_train=data_train,
                data_test=data_test,
                exog_test=exog_test,
                save_plots=True,
            )

    def predict(self, start_date: datetime):
        if self._model is None:
            raise UsageError("Model has not been trained.")
        self._check_dates(start=start_date)

        filename = "rocky_gully_near_6_features.csv"
        output_dir = project_config["output_dir"]
        data = load_data_from_csv(os.path.join(output_dir, filename))
        data, constant_data = extract_constant_features(data)

        freq = data.index.freq

        # TODO: assumes contiguous lags
        last_window_start = start_date - ((len(self._model.lags) + 1) * freq)

        last_window_end = start_date - freq

        last_window_data = data.loc[last_window_start:last_window_end]

        exog_data_all = pd.concat([constant_data] * len(data))
        exog_data_all.index = data.index
        exog_window_data = exog_data_all.loc[last_window_end + freq :]

        print(f"Last window: {last_window_start} to {last_window_end}")

        predictions = self._model.predict(
            last_window=last_window_data, exog=exog_window_data
        )[self._model.level]
        actuals = data.loc[predictions.index, self._model.level]

        self._plot_predictions_vs_actuals(
            predictions=predictions,
            actuals=actuals,
            historical_data=data.loc[
                last_window_start:last_window_end, self._model.level
            ],
            start_date=start_date,
            target_variable=self._model.level,
            save_plot=True,
        )
