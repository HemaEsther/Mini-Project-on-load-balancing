import numpy as np

def predict_and_select_balanced(df, arima_models, rf_models, gb_models, mse_stats,
                                current_time, num_nodes=10, eps=1e-6, alpha=0.6):
    preds = {}
    features = ["hour_sin","hour_cos","lag1","lag2","active_sessions","cpu","mem","latency"]

    for node in range(num_nodes):
        row = df[(df["node"]==node) & (df["time"]==current_time)].copy()
        if row.empty:
            row = df[df["node"]==node].iloc[[-1]]
        feat = row[features].iloc[0].values.reshape(1, -1)

        # Predictions
        try: rf_pred = float(rf_models[node].predict(feat))
        except: rf_pred = float(row["lag1"].values[0])
        try: gb_pred = float(gb_models[node].predict(feat))
        except: gb_pred = float(row["lag1"].values[0])
        try: arima_pred = float(arima_models[node].forecast(steps=1)[0])
        except: arima_pred = float(row["lag1"].values[0])

        # Blending
        mse_ar, mse_rf_, mse_gb_ = mse_stats[node]["mse_arima"]+eps, mse_stats[node]["mse_rf"]+eps, mse_stats[node]["mse_gb"]+eps
        w_ar, w_rf, w_gb = 1/mse_ar, 1/mse_rf_, 1/mse_gb_
        blended = (w_ar*arima_pred + w_rf*rf_pred + w_gb*gb_pred) / (w_ar + w_rf + w_gb)

        current_load = float(row["load"].values[0])
        score = alpha*current_load + (1-alpha)*blended

        preds[node] = {"arima":arima_pred,"rf":rf_pred,"gb":gb_pred,"blend":blended,
                       "current_load":current_load,"score":score}

    selected_node = min(preds.keys(), key=lambda x: preds[x]["score"])
    return selected_node, preds