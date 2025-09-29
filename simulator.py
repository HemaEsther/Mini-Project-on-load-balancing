import numpy as np
import pandas as pd
from dataset_generator import generate_synthetic_dataset, save_dataset_pdf
from model_trainer import train_initial_models
from predictor import predict_and_select_balanced
from job_assigner import assign_job_to_node

def run_simulation(num_nodes=10, timesteps=80, initial_train_end=50, verbose_steps=10):
    df = generate_synthetic_dataset(num_nodes, timesteps)
    df.to_csv("edge_dataset.csv", index=False)
    save_dataset_pdf(df, "edge_dataset.pdf", 25)

    arima_models, rf_models, gb_models, mse_stats = train_initial_models(df, num_nodes, initial_train_end)
    assignment_log = []

    for t in range(initial_train_end, timesteps):
        selected_node, preds = predict_and_select_balanced(df, arima_models, rf_models, gb_models, mse_stats, t, num_nodes)
        job_load = float(np.random.uniform(4.0, 15.0))
        df = assign_job_to_node(df, selected_node, t, job_load)

        entry = {
            "time": t, "selected_node": int(selected_node),
            "pred_blend": float(preds[selected_node]["blend"]),
            "pred_arima": float(preds[selected_node]["arima"]),
            "pred_rf": float(preds[selected_node]["rf"]),
            "pred_gb": float(preds[selected_node]["gb"]),
            "job_load": round(job_load,2),
            "new_load": float(df[(df.node==selected_node)&(df.time==t)].iloc[0]["load"])
        }
        assignment_log.append(entry)

        if len(assignment_log) <= verbose_steps:
            print(f"[t={t}] Assigned to Node {selected_node} | score={preds[selected_node]['score']:.2f} "
                  f"| job={job_load:.2f} | new_load={entry['new_load']:.2f}")

    assign_df = pd.DataFrame(assignment_log)
    df.to_csv("edge_dataset_after_assignments.csv", index=False)
    assign_df.to_csv("assignment_log.csv", index=False)
    return df, assign_df