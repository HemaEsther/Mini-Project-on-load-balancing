import pandas as pd
import numpy as np

def assign_job_to_node(df, node, time_step, job_load):
    mask = (df["node"]==node) & (df["time"]==time_step)
    if mask.sum() == 0:
        # Create a new row if it doesn't exist
        # Get the last known values for this node
        node_data = df[df["node"]==node]
        if len(node_data) > 0:
            last_row = node_data.iloc[-1].copy()
            last_row["time"] = time_step
            last_row["load"] = float(last_row["load"]) + job_load
            last_row["active_sessions"] = int(last_row["active_sessions"]) + int(round(job_load/3))
            last_row["cpu"] = float(np.clip(float(last_row["cpu"]) + job_load*0.6, 0, 100))
            last_row["mem"] = float(np.clip(float(last_row["mem"]) + job_load*0.4, 0, 100))
            # Update lag values
            last_row["lag1"] = last_row["load"]
            last_row["lag2"] = last_row["lag1"]
            df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)
        return df
    
    # Update existing row
    mask_idx = df[mask].index[0]
    df.loc[mask_idx, "load"] = float(df.loc[mask_idx, "load"]) + job_load
    df.loc[mask_idx, "active_sessions"] = int(df.loc[mask_idx, "active_sessions"]) + int(round(job_load/3))
    df.loc[mask_idx, "cpu"] = float(np.clip(float(df.loc[mask_idx, "cpu"]) + job_load*0.6, 0, 100))
    df.loc[mask_idx, "mem"] = float(np.clip(float(df.loc[mask_idx, "mem"]) + job_load*0.4, 0, 100))
    return df