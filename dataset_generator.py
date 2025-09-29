import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def generate_synthetic_dataset(num_nodes=10, timesteps=80, seed=42):
    np.random.seed(seed)
    rows = []
    for node in range(num_nodes):
        base = 30 + node * 2.5
        amp = 6 + (node % 4) * 2
        phase = (node / num_nodes) * 2 * np.pi
        for t in range(timesteps):
            daily = amp * np.sin(2 * np.pi * (t % 24) / 24.0 + phase)
            noise = np.random.normal(0, 3)
            spike = np.random.uniform(10, 35) if np.random.rand() < 0.01 else 0
            load = max(0.0, base + daily + noise + spike)
            active_sessions = max(1, int(round(load / 3.0 + np.random.normal(0,1))))
            cpu = min(100, max(0, load + np.random.normal(0,4)))
            mem = min(100, max(0, cpu * 0.6 + np.random.normal(0,3)))
            latency = max(1, 20 + (node * 0.6) + np.sin(t/12.0)*2 + np.random.normal(0,1))
            rows.append({
                "time": t, "node": int(node), "load": load,
                "active_sessions": active_sessions,
                "cpu": cpu, "mem": mem, "latency": latency
            })
    df = pd.DataFrame(rows).sort_values(["node","time"]).reset_index(drop=True)
    df["lag1"] = df.groupby("node")["load"].shift(1)
    df["lag2"] = df.groupby("node")["load"].shift(2)
    df["hour"] = df["time"] % 24
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df = df.fillna(method="bfill")
    return df

def save_dataset_pdf(df, out_pdf_path="edge_dataset.pdf", sample_rows=30):
    os.makedirs(os.path.dirname(out_pdf_path) or ".", exist_ok=True)
    pp = PdfPages(out_pdf_path)

    # Summary
    fig1, ax1 = plt.subplots(figsize=(8.27, 11.69))
    ax1.axis('off')
    summary = df.groupby("node")["load"].describe().round(2)
    ax1.text(0.01, 0.99, "Edge Dataset Summary (per node)\n\n" + summary.to_string(),
             va='top', ha='left', fontsize=8, family='monospace')
    pp.savefig(fig1, bbox_inches='tight')
    plt.close(fig1)

    # Sample
    sub = df.head(sample_rows)
    fig2, ax2 = plt.subplots(figsize=(11, 8.5))
    ax2.axis('off')
    table = ax2.table(cellText=sub.values, colLabels=sub.columns,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.2)
    ax2.set_title(f"Edge dataset sample (first {sample_rows} rows)", fontsize=10)
    pp.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

    pp.close()
    return out_pdf_path