from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import pandas as pd
import numpy as np

from joblib import load

from chronos import ChronosPipeline

from utils.arimax import Arimax
from utils.xgb import XGB
from utils.chronos import Chronos

# instantiate an instance of the app
app = FastAPI()

# read csv data
df = pd.read_csv("../data/processed/mastercard_processed.csv", parse_dates = True, index_col = ["Date"])

# class of data we want to send in post request
class PredData(BaseModel):
    days: int

# load the models
arimax = Arimax("../models/arimax_model.pkl")
xgb = XGB("../models/best_XGBRegressor_v1.pkl")
chronos_t5_small = Chronos("amazon/chronos-t5-small")

# post endpoints to get predictions from the models
@app.post("/arimax")
def arima_preds(data: PredData):
    # preprocess data into the form that the ARIMAX model accepts
    df_arima = arimax.preprocess_data(df)
    # use arimax to predict the future prices and confidence intervals
    predicted_prices, conf_int = arimax.predict(df_arima, data.days)
    return {"preds": predicted_prices, "conf_int": conf_int}

# post endpoint for XGB
@app.post("/XGB")
def xgb_preds(data: PredData):
    df_xgb = xgb.preprocess_data(df)
    # use XGB to predict the future prices
    predicted_prices = xgb.predict(df_xgb, data.days).tolist()
    return {"preds": predicted_prices}

# post endpoint for the chronos model
@app.post("/chronos")
def chronos_preds(data: PredData):
    low, median, high = chronos_t5_small.predict(df, data.days)
    low = low.tolist()
    median = median.tolist()
    high = high.tolist()
    return {"preds": median, "low": low, "high": high}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)