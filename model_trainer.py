import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def train_initial_models(df, num_nodes=10, train_end_time=50):
    arima_models, rf_models, gb_models, mse_stats = {}, {}, {}, {}
    features = ["hour_sin","hour_cos","lag1","lag2","active_sessions","cpu","mem","latency"]

    for node in range(num_nodes):
        node_df = df[(df["node"]==node) & (df["time"] < train_end_time)].copy()
        y = node_df["load"].values
        split = max(1, int(len(y)*0.8))
        train_y, val_y = y[:split], y[split:]

        # ARIMA
        try:
            arima_model = ARIMA(train_y, order=(2,0,2)).fit()
            arima_fore = arima_model.forecast(steps=len(val_y)) if len(val_y) > 0 else []
            mse_arima = float(mean_squared_error(val_y, arima_fore)) if len(val_y) > 0 else 1e-6
            arima_full = ARIMA(y, order=(2,0,2)).fit()
        except Exception:
            arima_full, mse_arima = None, 1e-6

        # Random Forest
        X = node_df[features].values; yvec = node_df["load"].values
        X_train, X_val, y_train, y_val = X[:split], X[split:], yvec[:split], yvec[split:]
        rf = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42)
        try:
            rf.fit(X_train, y_train)
            preds_rf = rf.predict(X_val) if len(X_val)>0 else rf.predict(X_train)
            mse_rf = float(mean_squared_error(y_val, preds_rf)) if len(X_val)>0 else 1e-6
        except Exception:
            rf, mse_rf = None, mse_arima

        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=60, max_depth=4, random_state=42)
        try:
            gb.fit(X_train, y_train)
            preds_gb = gb.predict(X_val) if len(X_val)>0 else gb.predict(X_train)
            mse_gb = float(mean_squared_error(y_val, preds_gb)) if len(X_val)>0 else 1e-6
        except Exception:
            gb, mse_gb = None, mse_rf

        arima_models[node], rf_models[node], gb_models[node] = arima_full, rf, gb
        mse_stats[node] = {"mse_arima": max(mse_arima,1e-6), "mse_rf": max(mse_rf,1e-6), "mse_gb": max(mse_gb,1e-6)}

    return arima_models, rf_models, gb_models, mse_stats