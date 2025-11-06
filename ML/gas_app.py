import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from api_eia import get_texas_gas_with_lags   # returns df_raw, df_lagged

st.set_page_config(
    page_title="⛽ Texas Gas Price Prediction",
    page_icon="⛽",
    layout="wide",
)
st.title("⛽ Texas Gas Price Prediction Dashboard")

# ---------------------
# Load and prepare data
# ---------------------
df_raw, df = get_texas_gas_with_lags(120)

df_raw = df_raw.sort_values("date").reset_index(drop=True)
df = df.sort_values("date").reset_index(drop=True)

if df_raw.empty or df.empty:
    st.error("❌ No data from EIA. Try again later.")
    st.stop()

# ---------------------
# Training dataset
# ---------------------
X = df[["lag1", "lag2", "lag3", "lag4", "lag5"]]
y = df["price"]

if len(X) < 2:
    st.error("Not enough rows for training. Increase history in get_texas_gas_with_lags().")
    st.stop()

# ---------------------
# Train/test split
# ---------------------
split = max(1, int(len(df) * 0.8))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

if X_train.empty or y_train.empty:
    st.error("Training data is empty. Increase history window in get_texas_gas_with_lags().")
    st.stop()

# ---------------------
# Train model
# ---------------------
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

df["prediction"] = model.predict(X)

# ---------------------
# Merge with raw data for display
# ---------------------
df_hist = df_raw.merge(df[["date", "prediction"]], on="date", how="left")

# If the last raw week has no prediction, compute it manually
if pd.isna(df_hist.loc[df_hist.index[-1], "prediction"]):
    latest_values = df.iloc[-1][["lag1", "lag2", "lag3", "lag4", "lag5"]].values.reshape(1, -1)
    df_hist.loc[df_hist.index[-1], "prediction"] = float(model.predict(latest_values)[0])

# ---------------------
# Clean display (remove missing)
# ---------------------
df_display = df_hist.dropna(subset=["prediction"]).reset_index(drop=True)

# ---------------------
# Next-week prediction
# ---------------------
last_raw_date = df_raw["date"].max()
next_week_date = last_raw_date + pd.Timedelta(weeks=1)
next_week_display = next_week_date.strftime("%Y/%m/%d")

latest_values = df.iloc[-1][["lag1", "lag2", "lag3", "lag4", "lag5"]].values.reshape(1, -1)
next_week_prediction = float(model.predict(latest_values)[0])
# ---------------------
col1, col2 = st.columns([1.2, 2])

with col1:
    st.markdown(
        f"<h2 style='color:green'>Predicted Price for Next Week ({next_week_display}): "
        f"${next_week_prediction:.3f} / gallon</h2>",
        unsafe_allow_html=True,
    )
    st.subheader("📅 Recent Actual Prices vs Predictions")
    st.dataframe(df_display.tail(12), use_container_width=True)

with col2:
    st.subheader("📊 Price Trends")
    st.line_chart(df_display.set_index("date")[["price", "prediction"]])

st.markdown("---")
st.caption("Data source: U.S. Energy Information Administration (EIA)")
