🛢️ Texas Gas Price Forecasting App
This project predicts Texas Regular Gasoline prices using real-world data from the U.S. Energy Information Administration (EIA).

⚙️ Features

Live Data Fetching from the EIA API

Data Cleaning & Transformation using pandas

Lag Feature Engineering for time-series modeling

Machine Learning Forecast using scikit-learn (Random Forest Regressor)

Interactive Visualization built with Streamlit

API Key Hidden via environment variables (.env / Streamlit Secrets)

🧠 Project Structure
texas_gas_forecast/

├── README.md

└── ML/
    ├── .env
    ├── .gitignore
    ├── requirements.txt
    ├── api_eia.py
    └── gas_app.py

🚀 How to Run Locally

1️⃣ Clone the repository
git clone https://github.com/Lindsay-Nguyen/gas_price_prediction_in_texas.git
cd gas_price_prediction_in_texas

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your EIA API key
Create a file named .env in the root folder:
EIA_API_KEY=your_api_key_here
(If running on Streamlit Cloud, add this key under Settings → Secrets instead.)

🧩 Example Usage
Run locally:
streamlit run gas_app.py

Or run API data fetch directly:
python api_eia.py

🧮 Model Overview

Algorithm: Random Forest Regressor
Features: Previous week prices (lag1–lag5)
Target: Next week’s price
Metrics: MAE, RMSE, R² Score

