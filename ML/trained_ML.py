import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- get gas data with lagged features ---
from api_eia import get_texas_gas_with_lags
df = get_texas_gas_with_lags(days_back=710)   # ~710 days (101 weeks)

# --- features: only past prices ---
X = df[["lag1", "lag2", "lag3", "lag4", "lag5"]]
y = df["target"]

# --- train/test split ---
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# fallback if test set too small
if len(X_test) == 0:
    X_train, X_test = X.iloc[:-1], X.iloc[-1:]
    y_train, y_test = y.iloc[:-1], y.iloc[-1:]

# --- train model ---
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# --- evaluate ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float("nan")

print("Model (gas-only) Evaluation")
print(f"MAE : {mae:.3f} $/gal")
print(f"RMSE: {rmse:.3f} $/gal")
print(f"R²  : {r2:.3f}" if np.isfinite(r2) else "R² : NA")

# --- predict next week ---
latest_row = X.iloc[-1:]
next_price = model.predict(latest_row)[0]
print(f"\nPredicted next week's TX Regular price: ${next_price:.3f}")

