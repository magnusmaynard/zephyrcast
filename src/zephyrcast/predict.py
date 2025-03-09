from skforecast.utils import load_forecaster
from zephyrcast import project_config
import os
import datetime
import ipdb
from click import UsageError
import pandas as pd
import matplotlib.pyplot as plt


from zephyrcast.utils import load_data_from_csv


def _find_latest_model_path():
    models_dir = project_config["models_dir"]
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]

    if not model_files:
        raise FileNotFoundError("No model files found in the models directory")

    if all("_" in f for f in model_files):
        model_files.sort(key=lambda x: x.split("_")[1:3], reverse=True)

    latest_model_path = os.path.join(models_dir, model_files[0])
    print(f"Loading latest model: {latest_model_path}")
    return latest_model_path


def _load_latest_model():
    latest_model_path = _find_latest_model_path()
    return load_forecaster(latest_model_path, verbose=True)


def _check_dates(model, start):
    training_start = model.training_range_[0]
    training_end = model.training_range_[1]

    if training_start <= start <= training_end:
        raise UsageError(
            f"The testing date {start.strftime('%Y-%m-%d %H:%M:%S')} is within the training range {training_start} -> {training_end}. Please choose a date outside this range."
        )


def plot_predictions_vs_actuals(
    predictions, actuals, historical_data, start_date, target_variable, save_plot=False
):
    # Calculate error metrics
    comparison_df = pd.DataFrame({"Predicted": predictions, "Actual": actuals})
    mae = (comparison_df["Predicted"] - comparison_df["Actual"]).abs().mean()
    rmse = ((comparison_df["Predicted"] - comparison_df["Actual"]) ** 2).mean() ** 0.5

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
        actuals.index, actuals.values, color="green", linewidth=2, label="Actual Values"
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
            plot_dir, f'forecast_comparison_{start_date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

    plt.tight_layout()
    plt.show()


def predict_live():
    raise NotImplementedError("Predict live is not implemented")


def predict_files(start_date: datetime):
    model = _load_latest_model()
    _check_dates(model=model, start=start_date)

    filename = "rocky_gully_near_6_features.csv"
    output_dir = project_config["output_dir"]
    data_all = load_data_from_csv(os.path.join(output_dir, filename))

    freq = data_all.index.freq

    # TODO: assumes contiguous lags
    last_window_start = start_date - ((len(model.lags) + 1) * freq)

    last_window_end = start_date - freq

    last_window_data = data_all.loc[last_window_start:last_window_end]

    print(f"Last window: {last_window_start} to {last_window_end}")

    predictions = model.predict(last_window=last_window_data)[ model.level]
    actuals = data_all.loc[predictions.index, model.level]

    plot_predictions_vs_actuals(
        predictions=predictions,
        actuals=actuals,
        historical_data=data_all.loc[
            last_window_start:last_window_end, model.level
        ],
        start_date=start_date,
        target_variable=model.level,
        save_plot=True,
    )
