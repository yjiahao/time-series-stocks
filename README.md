# Mastercard Stock Price Forecasting

This repository contains a time series analysis and forecasting project focused on Mastercard's stock prices. The goal is to analyze historical stock price data, generate insightful features, and build predictive models using ARIMAX and XGBoost.

# Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [License](#license)

## Project Overview

This project aims to:

Explore the trends and seasonality in Mastercard's stock price data.
Use statistical and machine learning models to forecast future stock prices.
Evaluate model performance using relevant metrics.

## Data Description

The dataset includes historical stock prices of Mastercard, with columns such as:

Date: The date of the observation.
Open/High/Low/Close: Stock prices at various points in the trading session.
Volume: The trading volume for the day.
Feature engineering was applied to create additional variables for improved prediction accuracy, including moving averages, returns, and lagged values.

## Exploratory Data Analysis

Key insights from EDA:

Trend and Seasonality: Identified using line plots of stock prices.
Autocorrelation and Partial Autocorrelation: Examined through ACF and PACF plots to determine the nature of time dependencies.

## Modeling
1. ARIMAX Model
Incorporated exogenous variables like trading volume and lagged features.
Optimized hyperparameters using grid search.
2. XGBoost Model
Engineered features such as:
Rolling averages
Lagged values
Volatility metrics
Trained and tuned the model using cross-validation.
Evaluation Metrics
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)

## Results

| Model             | RMSE before tuning | RMSE after tuning |
| :---------------- | :------: | :------: |
| ARIMAX            |   9.40   |     -    |
| XGBoost Regressor |   5.44   |   4.84   |

The XGBoost model outperformed ARIMAX in RMSE.

## Dependencies:

Install the dependencies with:

```bash
pip install -r requirements.txt
```
## How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/mastercard-stock-forecasting.git
cd mastercard-stock-forecasting
```

You can proceed to run the notebooks.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.