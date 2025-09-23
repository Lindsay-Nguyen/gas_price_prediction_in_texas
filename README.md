# gas_price_prediction_in_texas
⛽ Texas Gas Price Prediction

This project predicts next week’s regular gasoline price in Texas using data from the U.S. Energy Information Administration (EIA) and machine learning models. The pipeline includes live data collection, feature engineering (lags), model training, evaluation, and a simple frontend for visualization.
📌 Features
Live EIA API data
Automatically fetches Texas weekly regular gasoline prices from EIA

Feature engineering
Lag features (lag1–lag8) to capture past price history.

Target column = next week’s price.

Machine learning
Baseline model: Random Forest Regressor.
Evaluation metrics: MAE, RMSE, R².
Next-week price prediction.

Frontend (Streamlit)
Dashboard with last 10 weeks of data.
Line chart of historical & predicted values.
Live re-training on latest data.
