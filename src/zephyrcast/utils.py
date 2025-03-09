import pandas as pd

def load_data_from_csv(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)

    df["t_datetime"] = pd.to_datetime(df["t_datetime"])
    df.set_index("t_datetime", inplace=True)

    df = df.asfreq("10Min")
    df = df.sort_index()

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df

