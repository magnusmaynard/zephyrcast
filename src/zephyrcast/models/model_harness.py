from datetime import datetime
import os
from typing import Literal

from click import UsageError
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from zephyrcast.data.utils import get_train_test_data
from zephyrcast.models.baseline_model import BaselineModel
from zephyrcast.models.multivar_forecast_model import MultiVariantForecastModel
from zephyrcast.models.lstm_model import LSTMModel
from zephyrcast import project_config


class ModelHarness:
    def __init__(
        self, arch: Literal["baseline", "multivar", "seq2seq"], steps: int, target: str
    ):
        print("Initialising model...")
        self._steps = steps
        self._target = target
        if arch == "multivar":
            self._model = MultiVariantForecastModel(steps=steps, target=target)
        elif arch == "seq2seq":
            raise NotImplementedError("Seq2Seq not implemented")
            # self._model = Seq2SeqModel(steps=steps, target=target)
        elif arch == "lstm":
            self._model = LSTMModel(steps=steps, target=target)
        else:
            self._model = BaselineModel(steps=steps, target=target)

        self._data_filename = "rocky_gully_near_6_features.csv"
        self._data_split_date = datetime.strptime(
            "2025-02-07 23:59:00", "%Y-%m-%d %H:%M:%S"
        )
        self._data_train, self._data_test = get_train_test_data(
            filename=self._data_filename, train_split_date=self._data_split_date
        )
        
        # self._data_train[self._target].plot()
        # plt.show()

    def _plot_predictions(
        self, predictions: pd.DataFrame, actuals: pd.DataFrame, start_date: datetime, model_name: str
    ) -> None:
        context = 144  # 24 hours
        historical_data = self._data_test.loc[
            start_date - self._data_test.index.freq * context : start_date, self._target
        ]

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
            f"Forecast vs Actual Values for {self._target}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}\nModel: {model_name}"
        )
        plt.xlabel("Date")
        plt.ylabel(self._target)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Adjust x-axis date formatting
        plt.gcf().autofmt_xdate()

        # Save the plot if requested
        plot_dir = os.path.join(project_config.get("output_dir", ""), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir,
            f'prediction_{self._model.name}_{start_date.strftime("%Y%m%d_%H%M%S")}.png',
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

        plt.tight_layout()

    def _plot_evaluation_results(
        self, results_df: pd.DataFrame, start_date: datetime, end_date: datetime, model_name: str
    ) -> None:
        plt.figure(figsize=(14, 10))

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Plot 1: Predicted vs Actual
        axes[0].plot(
            results_df.index,
            results_df["Actual"],
            label="Actual",
            color="green",
            alpha=0.7,
        )
        axes[0].plot(
            results_df.index,
            results_df["Predicted"],
            label="Predicted",
            color="red",
        )
        axes[0].set_title(f"Predicted vs Actual Values for {self._target}")
        axes[0].set_ylabel(self._target)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Error over time
        axes[1].plot(results_df.index, results_df["Error"], color="blue")
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[1].set_title("Prediction Error Over Time")
        axes[1].set_ylabel("Error")
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Absolute Error over time with rolling average
        abs_error = results_df["AbsError"]
        rolling_mae = abs_error.rolling(
            window=48, min_periods=1
        ).mean()  # 48 points rolling window (adjust based on data frequency)

        axes[2].plot(
            results_df.index,
            abs_error,
            color="purple",
            alpha=0.5,
            label="Absolute Error",
        )
        axes[2].plot(
            results_df.index,
            rolling_mae,
            color="red",
            linewidth=2,
            label="Rolling MAE (48 periods)",
        )
        axes[2].set_title("Absolute Error and Rolling MAE")
        axes[2].set_ylabel("Absolute Error")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Customize the plot
        plt.suptitle(
            f'Evaluation Results: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}\nModel: {model_name}',
            fontsize=16,
        )

        # Adjust x-axis date formatting
        fig.autofmt_xdate()

        # Add overall error metrics as text
        plt.figtext(
            0.01,
            0.01,
            f"Overall Metrics:\n"
            f'MAE: {results_df["AbsError"].mean():.4f}\n'
            f'RMSE: {(results_df["Error"]**2).mean()**0.5:.4f}\n'
            f"Total Points: {len(results_df)}",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Save the plot
        plot_dir = os.path.join(project_config.get("output_dir", ""), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir,
            f'evaluation_{self._model.name}_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.png',
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Evaluation plot saved to: {plot_path}")

        plt.tight_layout()

    def train(self):
        print("Training...")
        self._model.train(data_train=self._data_train)

    def evaluate(self):
        print("Evaluating...")

        # if not self._model.is_trained:
        #     raise UsageError("Model must be trained before evaluation")

        context_size = self._model.window_size

        freq = self._data_test.index.freq

        start_eval_date = self._data_split_date + freq
        start_eval_date = start_eval_date + (context_size * freq)

        end_eval_date = self._data_test.index[-self._steps]# - freq * 3100

        all_predictions = []
        all_actuals = []

        prediction_window_size = self._steps * freq

        eval_dates = pd.date_range(start=start_eval_date, end=end_eval_date, freq=prediction_window_size)

        print(f"Backtesting from {start_eval_date} to {end_eval_date}")
        for eval_date in tqdm(eval_dates, desc="Evaluating", unit="window"):
            last_window_start = eval_date - self._model.window_size * freq
            last_window_end = eval_date
            last_window = self._data_test.loc[last_window_start:last_window_end]

            predictions = self._model.predict(last_window=last_window)
            actuals = self._data_test.loc[predictions.index, self._target]

            all_predictions.append(predictions)
            all_actuals.append(actuals)

        combined_predictions = pd.concat(all_predictions)
        combined_actuals = pd.concat(all_actuals)

        errors = combined_predictions - combined_actuals
        mae = errors.abs().mean()
        rmse = (errors**2).mean() ** 0.5
        mse = (errors**2).mean()

        results_df = pd.DataFrame(
            {
                "Predicted": combined_predictions,
                "Actual": combined_actuals,
                "Error": errors,
                "AbsError": errors.abs(),
            }
        )

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "MSE": mse,
            "NumPredictions": len(combined_predictions),
        }

        print(f"Evaluation metrics:")
        for metric, value in metrics.items():
            print(
                f"  {metric}: {value:.4f}"
                if isinstance(value, float)
                else f"  {metric}: {value}"
            )

        self._plot_evaluation_results(results_df, eval_dates[0], eval_dates[-1], model_name=self._model.name)

        return metrics

    def predict(self, date: datetime):
        print("Predicting...")
        # if not self._model.is_trained:
        #     raise UsageError("Model must be trained before predicting")

        if date <= self._data_split_date:
            raise UsageError(
                f"The testing date {date.strftime('%Y-%m-%d %H:%M:%S')} is within the training range {self._data_split_date}. Please choose a later date."
            )

        freq = self._data_test.index.freq
        last_window_start = date - (self._model.window_size * freq)
        last_window_end = date - freq
        last_window_data = self._data_test.loc[last_window_start:last_window_end]

        print(f"Last window: {last_window_start} to {last_window_end}")

        predictions = self._model.predict(last_window=last_window_data)
        print(predictions)

        actuals = self._data_test.loc[predictions.index, self._target]

        mse = np.mean((predictions.values - actuals.values) ** 2)
        print(f"MSE: {mse:.4f}")

        self._plot_predictions(
            predictions=predictions, actuals=actuals, start_date=date, model_name=self._model.name
        )
