import numpy as np
import pandas as pd
from data_loader import load_iot_dataset
from model_trainer import train_initial_models
from predictor import predict_and_select_balanced
from job_assigner import assign_job_to_node

def run_simulation(csv_path="iot_resource_allocation_dataset.csv", initial_train_end=50, verbose_steps=10):
    # Load data from CSV
    df = load_iot_dataset(csv_path)
    df.to_csv("edge_dataset.csv", index=False)
    
    # Get number of nodes and timesteps from the data
    num_nodes = df['node'].nunique()
    timesteps = df['time'].max() + 1
    
    # Adjust initial_train_end if it's too large
    if initial_train_end >= timesteps:
        initial_train_end = max(1, timesteps // 2)
        print(f"Warning: initial_train_end adjusted to {initial_train_end} (max timesteps: {timesteps})")

    arima_models, rf_models, gb_models, mse_stats = train_initial_models(df, num_nodes, initial_train_end)
    assignment_log = []

    for t in range(initial_train_end, timesteps):
        selected_node, preds = predict_and_select_balanced(df, arima_models, rf_models, gb_models, mse_stats, t, num_nodes)
        job_load = float(np.random.uniform(4.0, 15.0))
        df = assign_job_to_node(df, selected_node, t, job_load)

        # Get new load after assignment
        mask = (df["node"]==selected_node) & (df["time"]==t)
        if mask.sum() > 0:
            new_load = float(df[mask].iloc[0]["load"])
        else:
            # Fallback: use predicted load
            new_load = float(preds[selected_node]["blend"])
        
        entry = {
            "time": t, "selected_node": int(selected_node),
            "pred_blend": float(preds[selected_node]["blend"]),
            "pred_arima": float(preds[selected_node]["arima"]),
            "pred_rf": float(preds[selected_node]["rf"]),
            "pred_gb": float(preds[selected_node]["gb"]),
            "job_load": round(job_load,2),
            "new_load": new_load
        }
        assignment_log.append(entry)

        if len(assignment_log) <= verbose_steps:
            print(f"[t={t}] Assigned to Node {selected_node} | score={preds[selected_node]['score']:.2f} "
                  f"| job={job_load:.2f} | new_load={entry['new_load']:.2f}")

    assign_df = pd.DataFrame(assignment_log)
    df.to_csv("edge_dataset_after_assignments.csv", index=False)
    assign_df.to_csv("assignment_log.csv", index=False)
    return df, assign_df