from datetime import datetime
import pandas as pd
from zephyrcast import project_config
import os


def load_data_from_csv(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)

    df["t_datetime"] = pd.to_datetime(df["t_datetime"])
    df.set_index("t_datetime", inplace=True)

    df = df.asfreq("10Min")
    df = df.sort_index()

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def is_constant_feature(series, threshold=0.01):
    if series.nunique() <= 1:
        return True

    # Check if the series has very low variance compared to its mean
    std = series.std()
    mean = series.mean()

    # Avoid division by zero
    if mean == 0:
        return std < threshold

    # Check if the coefficient of variation is very small
    return abs(std / mean) < threshold


def extract_constant_features(data):
    constant_cols = []
    non_constant_cols = []

    for col in data.columns:
        if is_constant_feature(data[col]):
            constant_cols.append(col)
        else:
            non_constant_cols.append(col)

    print(f"Found {len(constant_cols)} constant features: {constant_cols}")

    # Create a dataframe with constant features (take first value for each)
    constant_features = {}
    for col in constant_cols:
        constant_features[col] = data[col].iloc[0]

    # Create a single-row dataframe with constant features
    constant_df = pd.DataFrame([constant_features])

    # Return dataframe without constant columns and the constant features
    return data[non_constant_cols], constant_df


def find_latest_model_path(prefix: str = "", suffix: str = ""):
    models_dir = project_config["models_dir"]
            
    model_files = [f for f in os.listdir(models_dir) if f.startswith(prefix) and f.endswith(suffix)]

    if not model_files:
        raise FileNotFoundError("No model files found in the models directory")

    if all("_" in f for f in model_files):
        model_files.sort(key=lambda x: x.split("_")[1:3], reverse=True)

    latest_model_path = os.path.join(models_dir, model_files[0])
    print(f"Loading latest model: {latest_model_path}")
    return latest_model_path

def get_train_test_data(
    filename: str,
    train_split_date: datetime,
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

    return data_train, data_test