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

from zephyrcast.data.utils import (
    extract_constant_features,
    find_latest_model_path,
)


class MultiVariantForecastModel:
    def __init__(self, steps: int, target: str):
        self._steps = steps
        self._target = target
        self._model = self._load_latest_model()

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

    def train(self, data_train: pd.DataFrame):
        data, data_const = extract_constant_features(data_train)
        data_exog = pd.concat([data_const] * len(data), ignore_index=True)
        data_exog.index = data.index

        ipdb.set_trace()
        window_features = RollingFeatures(stats=["mean", "sum"], window_sizes=[15, 15])
        series_features = list(data.drop(columns=self._target))
        lags = 24

        print(f"Forecast feature: {self._target}")
        print(f"Series features: {series_features}")
        print(f"Window features: {window_features}")
        print(f"Lags: {lags}")
        print(f"Exogenous features: {list(data_exog.columns)}")

        model = ForecasterDirectMultiVariate(
            regressor=LGBMRegressor(
                n_estimators=101, random_state=15926, max_depth=6, verbose=-1
            ),
            window_features=window_features,
            lags=lags,
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

    def predict(self, data_test: pd.DataFrame, start_date: datetime):
        data, data_const = extract_constant_features(data_test)
        data_exog = pd.concat([data_const] * len(data), ignore_index=True)
        data_exog.index = data.index
        freq = data.index.freq

        # TODO: assumes contiguous lags
        last_window_start = start_date - ((len(self._model.lags) + 1) * freq)

        last_window_end = start_date - freq

        last_window_data = data.loc[last_window_start:last_window_end]
        exog_window_data = data_exog.loc[last_window_end + freq :]

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

    def eval(self, data_test, save_plots):
        print("Testing...")
        output_dir = project_config["output_dir"]
        predictions = self._model.predict(exog=exog_test)
        actuals = data_test.loc[predictions.index][self._model.level]
        error_mse = mean_squared_error(y_true=actuals, y_pred=predictions)
        print(f"Test error (MSE): {error_mse}")

        if save_plots:
            context = 500
            fig, ax = plt.subplots(figsize=(6, 3))
            # data_train[-context:][self._model.level].plot(ax=ax, label="train")
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
