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


def _read_data(station_names: list[str]):
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
                row = {}
                for station_data in all_data:
                    station_name = station_data.get("name")
                    
                    if station_name in station_names:
                        station_index = station_names.index(station_name)
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


def _fill_missing(df, station_names: list[str]):
    df_new = df.copy()

    # List of features that should not change between rows.
    static_features = [
        "elev",
        "lat",
        "lon"
    ]
    
    for i, station in enumerate(station_names):
        for feature in static_features:
            feature_col = f"{i}_{feature}"
            
            if feature_col not in df_new.columns:
                continue
                
            valid_value = df_new[feature_col].dropna().iloc[0] if not df_new[feature_col].dropna().empty else None
            
            if valid_value is not None:
                df_new[feature_col] = df_new[feature_col].fillna(valid_value)

    return df_new


def _calculate_bearing(point1, point2):
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


def _add_time_features(df):
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


def _add_relative_features(df, nearby_station_count):
    df_features = df.copy()

    # Add wind cyclical features
    for index in range(nearby_station_count):
        station_id = index + 1
        bearing = df_features[f"{station_id}_wind_bearing"] 
        df_features[f"{station_id}_wind_dir_sin"] = np.sin(2 * np.pi * bearing / 360)
        df_features[f"{station_id}_wind_dir_cos"] = np.cos(2 * np.pi * bearing / 360)
    
    # Add elevation difference
    target_elevation = df_features["0_elev"].dropna().iloc[0]  # Assuming constant elevation for target
    
    for index in range(nearby_station_count):
        station_id = index + 1
        station_elevation = df_features[f"{station_id}_elev"].dropna().iloc[0]  # Assuming constant elevation for target

        df_features[f"{station_id}_elev_diff_from_target"] = station_elevation - target_elevation
    
    # Add distance and bearing between target and nearby stations
    target_lat = df_features["0_lat"].dropna().iloc[0]
    target_lon = df_features["0_lon"].dropna().iloc[0]
    target_coords = (target_lat, target_lon)
    
    for index in range(nearby_station_count):
        station_id = index + 1
        station_lat = df_features[f"{station_id}_lat"].dropna().iloc[0]
        station_lon = df_features[f"{station_id}_lon"].dropna().iloc[0]
        station_coords = (station_lat, station_lon)
        
        # Calculate distance in kilometers
        distance = geodesic(target_coords, station_coords).kilometers
        df_features[f"{station_id}_distance_from_target"] = distance
        
        # Calculate bearing from target to station
        bearing = _calculate_bearing(target_coords, station_coords)
        df_features[f"{station_id}_bearing_from_target"] = bearing
        
        # Add cyclical bearing features
        df_features[f"{station_id}_bearing_from_target_sin"] = np.sin(2 * np.pi * bearing / 360)
        df_features[f"{station_id}_bearing_from_target_cos"] = np.cos(2 * np.pi * bearing / 360)
        
        # Calculate if wind is coming from station direction (useful for predicting wind patterns)
        target_wind_bearing = df_features["0_wind_bearing"]
        station_to_target_bearing = (bearing + 180) % 360  # Reverse bearing
        
        # Calculate angular difference (0-180) between wind direction and station direction
        wind_from_station_angle = np.minimum(
            np.abs(target_wind_bearing - station_to_target_bearing), 
            360 - np.abs(target_wind_bearing - station_to_target_bearing)
        )
        df_features[f"{station_id}_wind_angle_from_target"] = wind_from_station_angle

    return df_features


def run():
    target_station = "Rocky Gully"
    nearby_stations = [
        # "Flightpark",
        # "Coronet Summit",
        # "Coronet Tandems",
        # "Queenstown Airport",
        # "Crown Terrace",
    ]
    
    print(f"Preparing data for '{target_station}', with respect to {nearby_stations}")

    station_names = [target_station, *nearby_stations]
    df = _read_data(station_names=station_names)
    df = _fill_missing(df=df, station_names=station_names)

    if df.empty:
        print(f"No data found", file=sys.stderr)
        sys.exit(1)
    
    print("Adding features...")
    df_features = df.copy()
    df_features = _add_time_features(df=df_features)
    df_features = _add_relative_features(df=df_features, nearby_station_count=len(nearby_stations))

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
    feature_filename = f"{target_station.replace(' ', '_').lower()}"
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