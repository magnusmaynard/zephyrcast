#!/usr/bin/env python3
import json
import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from zephyrcast import project_config
from datetime import datetime
from geopy.distance import geodesic


def prepare_data(target_station: str, nearby_stations: list[str]):
    data_dir = project_config["data_dir"]
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    json_files.sort()

    rows = []

    with tqdm(total=len(json_files), desc="Processing files", unit="file") as bar:
        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    all_data = json.load(f)

                # First find the target station data
                all_stations = [target_station, *nearby_stations]
                row = {}
                for station_data in all_data:
                    station_name = station_data.get("name")
                    
                    if station_name in all_stations:
                        station_index = all_stations.index(station_name)
                        wind_data = station_data.get("wind", {})
                        coords_data = station_data.get("coordinates", {})

                        row["t_stamp"] = station_data.get("timestamp")
                        row[f"{station_index}_wind_avg"] = wind_data.get("average")
                        row[f"{station_index}_wind_gust"] = wind_data.get("gust")
                        row[f"{station_index}_wind_bearing"] = wind_data.get("bearing")
                        row[f"{station_index}_temp"] = station_data.get("temperature")
                        row[f"{station_index}_elev"] = station_data.get("elevation")
                        row[f"{station_index}_lat"] = coords_data.get("lat")
                        row[f"{station_index}_lon"] = coords_data.get("lon")

                rows.append(row)

            except Exception as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)

            bar.update(1)

    return pd.DataFrame(rows)


def calculate_bearing(point1, point2):
    """
    Calculate the bearing between two points on the earth
    :param point1: (lat, lon) of point 1
    :param point2: (lat, lon) of point 2
    :return: bearing in degrees (0-360)
    """
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    
    bearing = np.degrees(np.arctan2(y, x))
    
    # Convert to compass bearing (0-360)
    bearing = (bearing + 360) % 360
    
    return bearing


def add_time_features(df):
    df_features = df.copy()
    
    # Convert timestamp to datetime and set as index
    df_features["t_stamp"] = pd.to_datetime(df_features["t_stamp"], unit='s')  # Unix timestamp
    df_features.set_index("t_stamp", inplace=True)
    
    # Add basic time features
    df_features["t_minute"] = df_features.index.minute
    df_features["t_hour"] = df_features.index.hour
    df_features["t_day"] = df_features.index.day
    df_features["t_month"] = df_features.index.month
    df_features["t_dayofyear"] = df_features.index.dayofyear
    
    # Add cyclical time features
    df_features["t_hour_sin"] = np.sin(2 * np.pi * df_features.index.hour / 24)
    df_features["t_hour_cos"] = np.cos(2 * np.pi * df_features.index.hour / 24)
    df_features["t_dayofyear_sin"] = np.sin(2 * np.pi * df_features.index.dayofyear / 365.25)
    df_features["t_dayofyear_cos"] = np.cos(2 * np.pi * df_features.index.dayofyear / 365.25)

    return df_features


def add_relative_features(df, target_station, nearby_stations):
    df_features = df.copy()
    
    # # Convert wind_bearing to cyclical features (handle NaN values)
    # if "wind_bearing" in df.columns:
    #     mask = df["wind_bearing"].notna()
    #     df_features["wind_dir_sin"] = np.nan
    #     df_features["wind_dir_cos"] = np.nan
        
    #     # Apply sin/cos calculations directly where wind_bearing is not NA
    #     wind_bearings = df.loc[mask, "wind_bearing"].values
    #     if len(wind_bearings) > 0:
    #         sin_values = np.sin(2 * np.pi * wind_bearings / 360)
    #         cos_values = np.cos(2 * np.pi * wind_bearings / 360)
            
    #         # Use the mask directly for assignment
    #         df_features.loc[mask, "wind_dir_sin"] = sin_values
    #         df_features.loc[mask, "wind_dir_cos"] = cos_values
    
    # Add elevation difference features if available
    target_elevation = df_features["0_elev"].iloc[0]  # Assuming constant elevation for target
    
    for index, station_name in enumerate(nearby_stations):
        station_id = index + 1
        station_elevation = df_features[f"{station_id}_elev"].iloc[0]  # Assuming constant elevation for target

        df_features[f"{station_id}_elev_diff"] = station_elevation - target_elevation
    
    # # Add distance and bearing features for each nearby station
    # target_coords = station_coordinates.get(target_station)
    
    # if target_coords:
    #     for idx, station in enumerate(nearby_stations):
    #         station_coords = station_coordinates.get(station)
    #         if station_coords:
    #             # Calculate distance in kilometers
    #             distance = geodesic(target_coords, station_coords).kilometers
    #             df_features[f"station_{idx}_distance"] = distance
                
    #             # Calculate bearing from target to station
    #             bearing = calculate_bearing(target_coords, station_coords)
    #             df_features[f"station_{idx}_bearing"] = bearing
                
    #             # Add cyclical bearing features
    #             df_features[f"station_{idx}_bearing_sin"] = np.sin(2 * np.pi * bearing / 360)
    #             df_features[f"station_{idx}_bearing_cos"] = np.cos(2 * np.pi * bearing / 360)
                
    #             # Add station wind_bearing cyclical features
    #             wind_bearing_col = f"station_{idx}_wind_bearing"
    #             if wind_bearing_col in df.columns:
    #                 # Handle NaN values
    #                 mask = df[wind_bearing_col].notna()
    #                 df_features[f"station_{idx}_wind_dir_sin"] = np.nan
    #                 df_features[f"station_{idx}_wind_dir_cos"] = np.nan
                    
    #                 if mask.any():
    #                     wind_bearings = df.loc[mask, wind_bearing_col].values
    #                     sin_values = np.sin(2 * np.pi * wind_bearings / 360)
    #                     cos_values = np.cos(2 * np.pi * wind_bearings / 360)
                        
    #                     df_features.loc[mask, f"station_{idx}_wind_dir_sin"] = sin_values
    #                     df_features.loc[mask, f"station_{idx}_wind_dir_cos"] = cos_values
                
    #             # Add wind vector components relative to bearing between stations
    #             wind_bearing_col = f"station_{idx}_wind_bearing"
    #             wind_avg_col = f"station_{idx}_wind_avg"
    #             if wind_bearing_col in df.columns and wind_avg_col in df.columns:
    #                 valid_mask = df[wind_bearing_col].notna() & df[wind_avg_col].notna()
    #                 df_features[f"station_{idx}_wind_along_bearing"] = np.nan
    #                 df_features[f"station_{idx}_wind_perp_bearing"] = np.nan
                    
    #                 if valid_mask.any():
    #                     wind_bearings = df.loc[valid_mask, wind_bearing_col].values
    #                     wind_speeds = df.loc[valid_mask, wind_avg_col].values
    #                     angle_diff = wind_bearings - bearing
                        
    #                     # Convert to radians for calculation
    #                     angle_diff_rad = np.radians(angle_diff)
                        
    #                     # Calculate components
    #                     along_values = wind_speeds * np.cos(angle_diff_rad)
    #                     perp_values = wind_speeds * np.sin(angle_diff_rad)
                        
    #                     df_features.loc[valid_mask, f"station_{idx}_wind_along_bearing"] = along_values
    #                     df_features.loc[valid_mask, f"station_{idx}_wind_perp_bearing"] = perp_values
    

    return df_features


def run():
    target_station = "Rocky Gully"
    nearby_stations = [
        "Flightpark",
        # "Coronet Summit",
        # "Coronet Tandems",
        # "Queenstown Airport",
        # "Crown Terrace",
    ]
    
    print(f"Preparing data for '{target_station}', with respect to {nearby_stations}")

    df = prepare_data(target_station=target_station, nearby_stations=nearby_stations)

    if df.empty:
        print(f"No data found", file=sys.stderr)
        sys.exit(1)
    
    print("Adding features...")
    df_features = df.copy()
    df_features = add_time_features(df_features)
    df_features = add_relative_features(df_features, target_station, nearby_stations)

    # Sort for easier viewing
    df_features = df_features.reindex(sorted(df_features.columns), axis=1)

    output_dir = project_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save original data
    original_filename = f"{target_station.replace(' ', '_').lower()}"
    if len(nearby_stations) > 0:
        original_filename += f"_{len(nearby_stations)}"
    original_filename += "_raw.csv"
    original_output_file = os.path.join(output_dir, original_filename)
    
    # Save featured data
    feature_filename = f"{target_station.replace('# ', '_').lower()}"
    if len(nearby_stations) > 0:
        feature_filename += f"_{len(nearby_stations)}"
    feature_filename += "_features.csv"
    feature_output_file = os.path.join(output_dir, feature_filename)

    # Save both dataframes
    df.to_csv(original_output_file, index=False)
    df_features.to_csv(feature_output_file, index=True)

    print(f"Total results processed: {len(df)}")
    print(f"Original data saved to '{original_output_file}'")
    print(f"Featured data saved to '{feature_output_file}'")
    
    return df, df_features