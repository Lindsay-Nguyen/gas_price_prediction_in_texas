from datetime import datetime, timedelta
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# Get your EIA API key safely from .env file
API_KEY = os.getenv("EIA_API_KEY")

if not API_KEY:
    raise ValueError(" Missing EIA_API_KEY. Please set it in your .env file.")

def get_texas_gas_with_lags(days_back=2000):
    """
    Fetch TX Regular Gasoline weekly prices from EIA.
    Returns two DataFrames:
      - df_raw: clean weekly series (date, price)
      - df_lagged: same with lag1–lag8 + target (safe for ML)
    """
    start = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = (
        "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
        "?frequency=weekly"
        "&data[0]=value"
        "&facets[duoarea][]=STX"   # Texas
        f"&start={start}"
        "&sort[0][column]=period"
        "&sort[0][direction]=asc"
        f"&api_key={API_KEY}"
    )

    r = requests.get(url)
    data = r.json()
    if "response" not in data:
        raise ValueError(f"EIA API error: {data}")

    df = pd.DataFrame(data["response"]["data"])

    if df.empty:
        raise ValueError("No data returned from EIA API. Try increasing days_back.")

    # --- filter only Regular Gasoline ---
    df = df[df["product-name"].str.contains("Regular", case=False)]
    if df.empty:
        raise ValueError(" No 'Regular Gasoline' found. Check available products above.")

    # Keep only date + price
    df = df[["period", "value"]].rename(columns={"period": "date", "value": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)

    # Deduplicate
    df = df.groupby("date", as_index=False).agg({"price": "mean"}).sort_values("date")

    # ---------------------
    # df_raw: clean weekly data
    # ---------------------
    df_raw = df.reset_index(drop=True)

    # ---------------------
    # df_lagged: add lag features (1–8 weeks)
    # ---------------------
    df_lagged = df.copy()
    for lag in range(1, 9):
        df_lagged[f"lag{lag}"] = df_lagged["price"].shift(lag)

    df_lagged["target"] = df_lagged["price"].shift(-1)
    df_lagged = df_lagged.dropna().reset_index(drop=True)

    return df_raw, df_lagged


if __name__ == "__main__":
    df_raw, df_lagged = get_texas_gas_with_lags(2000)
    print("RAW (keeps last week):")
    print(df_raw.tail(5))
    print("\nLAGGED (training, drops last week):")
    print(df_lagged.tail(5))
