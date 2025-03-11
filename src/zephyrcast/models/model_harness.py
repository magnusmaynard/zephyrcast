from datetime import datetime
import os
from typing import Literal

from click import UsageError
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from zephyrcast.data.utils import get_train_test_data
from zephyrcast.models.baseline_model import BaselineModel
from zephyrcast.models.multivar_forecast_model import MultiVariantForecastModel
from zephyrcast.models.seq2seq_model import Seq2SeqModel
from zephyrcast import project_config


class ModelHarness:
    def __init__(
        self, arch: Literal["baseline", "multivar", "seq2seq"], steps: int, target: str
    ):
        print("Initialising model")
        self._steps = steps
        self._target = target
        if arch == "multivar":
            self._model = MultiVariantForecastModel(steps=steps, target=target)
        elif arch == "seq2seq":
            self._model = Seq2SeqModel(steps=steps, target=target)
        else:
            self._model = BaselineModel(steps=steps, target=target)

        self._data_filename = "rocky_gully_near_6_features.csv"
        self._data_split_date = datetime.strptime(
            "2025-02-07 23:59:00", "%Y-%m-%d %H:%M:%S"
        )
        self._data_train, self._data_test = get_train_test_data(
            filename=self._data_filename, train_split_date=self._data_split_date
        )

    def _plot_predictions_vs_actuals(
        self, predictions: pd.DataFrame, actuals: pd.DataFrame, start_date: datetime
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
            f"Forecast vs Actual Values for {self._target}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}"
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
            f'forecast_comparison_{start_date.strftime("%Y%m%d_%H%M%S")}.png',
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

        plt.tight_layout()

    def train(self):
        print("Training...")
        self._model.train(data_train=self._data_train)

    def evaluate(self):
        print("Evaluating...")
        pass

    def predict(self, date: datetime):
        print("Predicting...")
        if date <= self._data_split_date:
            raise UsageError(
                f"The testing date {date.strftime('%Y-%m-%d %H:%M:%S')} is within the training range {self._data_split_date}. Please choose a later date."
            )

        freq = self._data_test.index.freq
        last_window_start = date - (self._model.context * freq)
        last_window_end = date - freq
        last_window_data = self._data_test.loc[last_window_start:last_window_end]

        print(f"Last window: {last_window_start} to {last_window_end}")

        predictions = self._model.predict(last_window=last_window_data)
        print(predictions)

        actuals = self._data_test.loc[predictions.index, self._target]

        mse = np.mean((predictions.values - actuals.values) ** 2)
        print(f"MSE: {mse:.4f}")

        self._plot_predictions_vs_actuals(
            predictions=predictions, actuals=actuals, start_date=date
        )
