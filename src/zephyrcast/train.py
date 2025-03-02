import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import random
from zephyrcast import project_config



def _load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    df["t_stamp"] = pd.to_datetime(df["t_stamp"])
    df.set_index("t_stamp", inplace=True)

    # Sort by datetime index
    df.sort_index(inplace=True)

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def _add_lag_features(df, target_cols: list[str], future_steps: int, lag_steps: int):
    """Create features and targets with a simplified approach"""
    # Create a copy for feature engineering
    df_features = df.copy()

    # Create lag features (configurable lookback)
    for col in target_cols:
        # Add basic lag features
        for lag in range(1, lag_steps + 1):
            df_features[f"{col}_lag_{lag}"] = df_features[col].shift(lag)
            
        # # Add rolling mean features
        # df_features[f"{col}_ma_3"] = df_features[col].rolling(3).mean()
        # df_features[f"{col}_ma_6"] = df_features[col].rolling(6).mean()
        
        # # Add rolling standard deviation
        # df_features[f"{col}_std_3"] = df_features[col].rolling(3).std()
        
        # # Add rate of change features
        # df_features[f"{col}_roc_1"] = df_features[col].pct_change(1)
        # df_features[f"{col}_roc_3"] = df_features[col].pct_change(3)

    # Create future targets
    for col in target_cols:
        for step in range(1, future_steps + 1):
            df_features[f"{col}_future_{step}"] = df_features[col].shift(-step)

    # Drop rows with missing values
    df_clean = df_features.dropna()

    print(f"After feature preparation: {len(df_clean)} rows")
    return df_clean

def _split_features_targets(df, target_cols, future_steps):
    """Split data into features and targets"""
    # Identify target columns
    future_target_cols = []
    for col in target_cols:
        for step in range(1, future_steps + 1):
            future_target_cols.append(f"{col}_future_{step}")

    # Extract targets as a dictionary
    targets = {}
    for col in target_cols:
        for step in range(1, future_steps + 1):
            target_name = f"{col}_future_{step}"
            targets[(col, step)] = df[target_name]

    # Extract feature columns (all except future targets)
    feature_cols = [col for col in df.columns if col not in future_target_cols]
    X = df[feature_cols]

    print(f"Features: {len(feature_cols)} columns")
    print(f"Targets: {len(targets)} columns")

    return X, targets, feature_cols


def _train_models(X, targets, feature_cols):
    print("Training models...")

    models = {}
    metrics = {}

    # Create train/test mask based on day of month
    # Every 7th, 14th, 21st, and 28th day used for testing
    is_test_day = [d.day % 7 == 0 for d in X.index]

    # Split data
    X_train = X[~np.array(is_test_day)]
    X_test = X[np.array(is_test_day)]

    print(f"Training data: {len(X_train)} samples")
    print(f"Testing data: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # For each target variable and forecast horizon
    for (col, step), y in targets.items():
        print(f"Training model for {col} {step} steps ahead:")

        # Split the target accordingly
        y_train = y[~np.array(is_test_day)]
        y_test = y[np.array(is_test_day)]

        # Train the model with simplified parameters
        model = xgb.XGBRegressor(
            n_estimators=100,  # Increased number of trees
            learning_rate=0.05,  # Reduced learning rate for better precision
            max_depth=5,  # Slightly increased complexity
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,  # L1 regularization
            reg_lambda=0.01,  # L2 regularization
            min_child_weight=3,  # More conservative splits
            random_state=42,
        )

        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Store results
        models[(col, step)] = model
        metrics[(col, step)] = {"mae": mae, "rmse": rmse, "r2": r2}

        # Print metrics
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return models, metrics, scaler, X_train, X_test


def _visualize_results(X, targets, models, metrics, scaler):
    """Visualize model performance and feature importance"""

    # Scale features for prediction
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # Set up plots for each target and forecast horizon
    target_cols = list(set([col for col, _ in targets.keys()]))
    max_step = max([step for _, step in targets.keys()])

    fig, axes = plt.subplots(len(target_cols), max_step, figsize=(15, 10))
    if len(target_cols) == 1:
        axes = np.array([axes])

    # Plot predictions vs. actuals
    for i, col in enumerate(target_cols):
        for step in range(1, max_step + 1):
            j = step - 1

            model = models[(col, step)]
            predictions = model.predict(X_scaled)
            actuals = targets[(col, step)]

            # Use only the last 100 data points for clarity
            sample_size = min(100, len(predictions))
            sample_idx = -sample_size

            ax = axes[i, j]
            ax.plot(
                X.index[sample_idx:],
                actuals.iloc[sample_idx:],
                label="Actual",
                color="blue",
            )
            ax.plot(
                X.index[sample_idx:],
                predictions[sample_idx:],
                label="Predicted",
                color="red",
                linestyle="--",
            )

            ax.set_title(f"{col} (+{step*10} min)")
            if i == len(target_cols) - 1:
                ax.set_xlabel("Date")
            if j == 0:
                ax.set_ylabel(col)

            # Add metrics to the plot
            metrics_text = f"MAE: {metrics[(col, step)]['mae']:.2f}\nRMSE: {metrics[(col, step)]['rmse']:.2f}"
            ax.text(
                0.05,
                0.95,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
            )

            # Add legend to first plot only
            if i == 0 and j == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(project_config["output_dir"], "forecast_visualization.png"))


def _visualize_future_predictions(
    X_test, df_raw, models, metrics, scaler, target_cols, future_steps, lag_steps, num_samples=5
):
    """Visualize predictions for random test samples with error bars, including past data points used for prediction"""
    # Get available test indices
    test_indices = list(X_test.index)

    # Select random samples (making sure we have enough future data)
    valid_indices = [
        idx
        for idx in test_indices
        if idx + pd.Timedelta(minutes=10 * future_steps) <= df_raw.index.max()
    ]

    if len(valid_indices) < num_samples:
        print(f"Warning: Only {len(valid_indices)} valid samples available")
        num_samples = len(valid_indices)

    if num_samples == 0:
        print("No valid samples available for visualization")
        return

    random.seed(42)  # For reproducible results
    sample_indices = random.sample(valid_indices, num_samples)

    # Create separate plots for each sample and target
    for i, start_idx in enumerate(sample_indices):

        # For each target variable, create a separate figure
        for j, col in enumerate(target_cols):
            plt.figure(figsize=(10, 6))

            # Get feature row and scale
            feature_row = X_test.loc[start_idx:start_idx].copy()
            feature_row_scaled = pd.DataFrame(
                scaler.transform(feature_row),
                columns=feature_row.columns,
                index=feature_row.index,
            )

            # Get past data points used for prediction (lookback)
            past_times = []
            past_values = []

            # We use LOOKBACK_STEPS for the past data
            for step in range(1, lag_steps + 1):
                past_time = start_idx - pd.Timedelta(minutes=10 * step)
                if past_time in df_raw.index:
                    past_times.insert(
                        0, past_time
                    )  # Insert at beginning to maintain chronological order
                    past_values.insert(0, df_raw.loc[past_time, col])

            # Get actual future values
            actual_times = []
            actual_values = []

            for step in range(1, future_steps + 1):
                future_time = start_idx + pd.Timedelta(minutes=10 * step)
                if future_time in df_raw.index:
                    actual_times.append(future_time)
                    actual_values.append(df_raw.loc[future_time, col])

            # Get predictions
            pred_times = []
            pred_values = []
            pred_errors = []

            for step in range(1, future_steps + 1):
                model = models[(col, step)]
                prediction = model.predict(feature_row_scaled)[0]
                error = metrics[(col, step)]["rmse"]

                future_time = start_idx + pd.Timedelta(minutes=10 * step)
                pred_times.append(future_time)
                pred_values.append(prediction)
                pred_errors.append(error)

            # Plot past data points (used for prediction)
            all_past_times = past_times + [start_idx]
            all_past_values = past_values + [df_raw.loc[start_idx, col]]
            plt.plot(all_past_times, all_past_values, "g-o", label="Past data (inputs)")

            # Shade the prediction area
            plt.axvspan(start_idx, pred_times[-1], alpha=0.1, color="gray")

            # Plot actuals
            if actual_times:
                all_times = [start_idx] + actual_times
                all_values = [df_raw.loc[start_idx, col]] + actual_values
                plt.plot(all_times, all_values, "b-o", label="Actual future")

            # Plot predictions without error bars
            plt.plot(pred_times, pred_values, "r--o", label="Prediction")

            # Add vertical line at prediction start
            plt.axvline(x=start_idx, color="k", linestyle="--", alpha=0.5)
            plt.text(
                start_idx,
                plt.ylim()[0],
                "Prediction start",
                rotation=90,
                verticalalignment="bottom",
            )

            # Add RMSE to the title instead of error bars
            avg_rmse = sum(pred_errors) / len(pred_errors)
            plt.title(
                f"Sample {i+1}: {col} prediction using {lag_steps} past readings (Avg RMSE: {avg_rmse:.2f})"
            )
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()

            # Save individual plot
            plt.savefig(
                os.path.join(project_config["output_dir"], f"future_pred_sample{i+1}_{col}.png")
            )
            plt.close()


# def _save_models(models, scaler, feature_cols):
#     model_dir = project_config["models_dir"]
#     """Save trained models and related data"""
#     os.makedirs(model_dir, exist_ok=True)

#     # Save models
#     for (col, step), model in models.items():
#         filename = os.path.join(model_dir, f"{col}_step{step}_model.joblib")
#         joblib.dump(model, filename)

#     # Save scaler
#     scaler_filename = os.path.join(model_dir, "feature_scaler.joblib")
#     joblib.dump(scaler, scaler_filename)

#     # Save feature columns
#     feature_cols_filename = os.path.join(model_dir, "feature_columns.joblib")
#     joblib.dump(feature_cols, feature_cols_filename)

#     # Save configuration
#     config = {
#         "target_columns": TARGET_COLUMNS,
#         "future_steps": FUTURE_STEPS,
#         "lookback_steps": LOOKBACK_STEPS,
#     }

#     config_filename = os.path.join(model_dir, "model_config.joblib")
#     joblib.dump(config, config_filename)


def run():
    target_cols = ["0_wind_avg", "0_wind_gust", "0_wind_bearing", "0_temp"]
    future_steps = 3  # 30 minutes (3 x 10 minute intervals)
    lag_steps = 6  # Use 6 previous readings (60 minutes)
    data = "output/rocky_gully_1_features.csv"

    df = _load_data_from_csv(data)

    df_lagged = _add_lag_features(df, target_cols=target_cols, future_steps=future_steps, lag_steps=lag_steps)

    X, targets, feature_cols = _split_features_targets(
        df_lagged, target_cols=target_cols, future_steps=future_steps
    )

    models, metrics, scaler, X_train, X_test = _train_models(X, targets, feature_cols)

    _visualize_results(X, targets, models, metrics, scaler)
    _visualize_future_predictions(
        X_test,
        df,
        models,
        metrics,
        scaler,
        target_cols,
        future_steps,
        lag_steps,
        3,
    )

    # _save_models(models, scaler, feature_cols)

    print("Training complete")
