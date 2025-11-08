import numpy as np
import pandas as pd
from datetime import datetime

def load_iot_dataset(csv_path="iot_resource_allocation_dataset.csv"):
    """
    Load and transform IoT resource allocation dataset to match expected format.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with columns: time, node, load, active_sessions, cpu, mem, latency, 
                                lag1, lag2, hour, hour_sin, hour_cos
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Extract device number from Device_ID and map to consecutive node IDs (0, 1, 2, ...)
    device_ids = df['Device_ID'].str.extract(r'(\d+)').astype(int).squeeze()
    unique_devices = sorted(device_ids.unique())
    device_to_node = {dev: idx for idx, dev in enumerate(unique_devices)}
    df['node'] = device_ids.map(device_to_node)
    
    # Convert Timestamp to time steps
    df['timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
    unique_timestamps = sorted(df['timestamp'].unique())
    time_map = {ts: idx for idx, ts in enumerate(unique_timestamps)}
    df['time'] = df['timestamp'].map(time_map)
    
    # Map columns
    df['cpu'] = df['CPU_Usage(%)'].clip(0, 100)
    # Normalize memory to 0-100 range (assuming max is around 8000 MB)
    max_mem = df['Memory_Usage(MB)'].max()
    df['mem'] = (df['Memory_Usage(MB)'] / max_mem * 100).clip(0, 100)
    df['latency'] = df['Network_Latency(ms)']
    
    # Create load from CPU usage (can be adjusted based on requirements)
    # Using CPU as base load indicator
    df['load'] = df['cpu'].copy()
    
    # Create active_sessions (derive from workload or use a simple formula)
    # Using a simple formula based on CPU and memory
    df['active_sessions'] = (df['cpu'] / 10 + df['mem'] / 20).clip(1, 100).astype(int)
    
    # Sort by node and time for lag calculations
    df = df.sort_values(['node', 'time']).reset_index(drop=True)
    
    # Create lag features
    df['lag1'] = df.groupby('node')['load'].shift(1)
    df['lag2'] = df.groupby('node')['load'].shift(2)
    
    # Extract hour from timestamp
    df['hour'] = df['timestamp'].dt.hour
    
    # Create hour features (sin/cos encoding)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Fill NaN values in lag columns (forward fill, then backward fill)
    df['lag1'] = df.groupby('node')['lag1'].bfill()
    df['lag2'] = df.groupby('node')['lag2'].bfill()
    df['lag1'] = df.groupby('node')['lag1'].ffill()
    df['lag2'] = df.groupby('node')['lag2'].ffill()
    # If still NaN, fill with current load
    df['lag1'] = df['lag1'].fillna(df['load'])
    df['lag2'] = df['lag2'].fillna(df['load'])
    
    # Select and reorder columns
    result_df = df[['time', 'node', 'load', 'active_sessions', 'cpu', 'mem', 'latency', 
                    'lag1', 'lag2', 'hour', 'hour_sin', 'hour_cos']].copy()
    
    return result_df

