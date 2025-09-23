# ⛽ gas_price_prediction_in_texas

This project predicts next week’s **regular gasoline price in Texas** using live data from the [EIA API](https://www.eia.gov/) and machine learning models.  
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

## 📊 Model Performance (Gas-only, Random Forest)

Using ~710 days (~101 weeks) of Texas Regular Gasoline prices:

- **MAE (Mean Absolute Error):** ~0.078 $/gal  
- **RMSE (Root Mean Squared Error):** ~0.088 $/gal  
- **R² (Explained Variance):** -2.61 (baseline, limited by only 3 lag features)  
- **Prediction Horizon:** Next week’s gasoline price  
- **Predicted price example:** $2.756/gal (for week of 2025-09-22)  

⚡ *Interpretation:*  
The model predicts within ~8 cents per gallon on average.  
While R² is negative (showing that weekly gas prices are highly volatile), this sets a **baseline** and motivates adding **longer lag features + sentiment signals from news** to improve performance.  
