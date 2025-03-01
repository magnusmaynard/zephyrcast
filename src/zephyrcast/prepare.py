#!/usr/bin/env python3
import json
import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

def prepare_data(station_name="Rocky Gully"):
    data_dir = "data"
    json_files = glob.glob(os.path.join(data_dir, "*.json"))    
    json_files.sort()
    
    records = []
    
    with tqdm(total=len(json_files), desc="Processing files", unit="file") as pbar:
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    all_data = json.load(f)
                
                for station_data in all_data:
                    if station_data.get("name") == station_name:
                        wind_data = station_data.get("wind", {})
                        
                        record = {
                            "timestamp": station_data.get("timestamp"),
                            "wind_avg": wind_data.get("average"),
                            "wind_gust": wind_data.get("gust"),
                            "wind_bearing": wind_data.get("bearing"),
                            "temperature": station_data.get("temperature")
                        }
                        records.append(record)
                        break
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)
            
            pbar.update(1)
    
    return pd.DataFrame(records)

def run():
    station_name = "Rocky Gully"
    print(f"Filtering data for station: {station_name}")
    
    df = prepare_data(station_name)
    
    if df.empty:
        print(f"No data found", file=sys.stderr)
        sys.exit(1)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{station_name.replace(' ', '_').lower()}_data.csv")
    
    df.to_csv(output_file, index=False)
    
    print(f"Total results processed: {len(df)}")
    print(f"Saved to '{output_file}'")
