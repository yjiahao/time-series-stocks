from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np

from joblib import load

from chronos import ChronosPipeline

from utils.arimax import preprocess_arima_data, arimax_predict
from utils.xgb import preprocess_xgb_data, xgb_predict
from utils.chronos import chronos_predict

# instantiate an instance of the app
app = FastAPI()

# read csv data
df = pd.read_csv("../data/processed/mastercard_processed.csv", parse_dates = True, index_col = ["Date"])

# class of data we want to send in post request
class PredData(BaseModel):
    days: int

# load the models
ARIMAX = load("../models/arimax_model.pkl")
XGB = load("../models/best_XGBRegressor_v1.pkl")
CHRONOS_T5_SMALL = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")

# post endpoints to get predictions from the models
@app.post("/arimax")
def arima_preds(data: PredData):
    # preprocess data into the form that the ARIMAX model accepts
    df_arima = preprocess_arima_data(df)
    # use arimax to predict the future prices and confidence intervals
    predicted_prices, conf_int = arimax_predict(ARIMAX, df_arima, data.days)
    return {"preds": predicted_prices, "conf_int": conf_int}

# post endpoint for XGB
@app.post("/XGB")
def xgb_preds(data: PredData):
    df_xgb = preprocess_xgb_data(df)
    # use XGB to predict the future prices
    predicted_prices = xgb_predict(XGB, df_xgb, data.days).tolist()
    return {"preds": predicted_prices}

# post endpoint for the chronos model
@app.post("/chronos")
def chronos_preds(data: PredData):
    low, median, high = chronos_predict(CHRONOS_T5_SMALL, df, data.days)
    low = low.tolist()
    median = median.tolist()
    high = high.tolist()
    return {"preds": median, "low": low, "high": high}