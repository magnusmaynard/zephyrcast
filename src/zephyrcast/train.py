import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skforecast.recursive import ForecasterRecursive
import math
import os
from zephyrcast import project_config


def _load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    df["t_stamp"] = pd.to_datetime(df["t_stamp"])
    df.set_index("t_stamp", inplace=True)

    df = df.asfreq("10Min")
    df = df.sort_index()

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def run():
    output_dir = project_config["output_dir"]
    data = _load_data_from_csv(os.path.join(output_dir, "rocky_gully_features.csv"))

    # Select features and target
    target = "0_temp"

    # Split data: Last month for testing
    split_date = data.index[-1] - pd.DateOffset(months=1)
    data_train = data[data.index < split_date]
    data_test = data[data.index >= split_date]

    # Train forecaster
    lags = 24
    steps = 6
    regressor = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=123)
    forecaster = ForecasterRecursive(regressor=regressor, lags=lags)
    print("Fitting...")
    forecaster.fit(y=data_train[target])

    # Make predictions
    print("Predicting...")
    last_window = data_train[target][-lags:]  # Get the last window from training data

    # Create a date range for predictions
    prediction_index = pd.date_range(
        start=data_train.index[-1] + pd.Timedelta(minutes=10),
        periods=steps,
        freq="10min",
    )

    # Get actual values for the same time period
    actual_values = data_test[target].iloc[:steps].values

    # Make predictions
    predictions = forecaster.predict(steps=steps, last_window=last_window)
    predictions.index = prediction_index  # Assign the proper datetime index

    # Evaluate model
    mse = mean_squared_error(y_true=actual_values, y_pred=predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true=actual_values, y_pred=predictions)
    r2 = r2_score(y_true=actual_values, y_pred=predictions)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Assign proper datetime index to last_window
    last_window_index = pd.date_range(
        end=data_train.index[-1], periods=lags, freq="10min"
    )

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot actual test data with more data points for context
    context_steps = min(48, len(data_test))  # Show up to 48 points for context
    plt.plot(
        data_test[target].iloc[:context_steps].index,
        data_test[target].iloc[:context_steps].values,
        label="Actual Values",
        color="blue",
    )

    # Plot predictions
    plt.plot(
        predictions.index,
        predictions.values,
        label="Predictions",
        color="red",
        linestyle="--",
        marker="o",
    )

    # Plot last window used for prediction
    plt.plot(
        last_window_index,
        last_window.values,
        label="Last Window (Input)",
        color="green",
        linestyle="dotted",
        marker="x",
    )

    # Highlight the prediction area
    plt.axvspan(prediction_index[0], prediction_index[-1], color="red", alpha=0.1)

    plt.title(f"10-Minute Temperature Forecast\nRMSE: {rmse:.4f}°C")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "forecast_results.png"), dpi=300)
