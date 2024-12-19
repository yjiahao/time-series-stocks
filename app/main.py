from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np

from joblib import load

from chronos import ChronosPipeline

from utils import preprocess_arima_data, arimax_predict

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

# post requests to get predictions from the models
@app.post("/arimax")
def arima_preds(data: PredData):
    # preprocess data into the form that the ARIMAX model accepts
    df_arima = preprocess_arima_data(df)
    # use arimax to predict the future prices and confidence intervals
    predicted_prices, conf_int = arimax_predict(ARIMAX, df_arima, data.days)
    return {"preds": predicted_prices, "conf_int": conf_int}