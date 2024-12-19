from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np

from joblib import load

from chronos import ChronosPipeline

# instantiate an instance of the app
app = FastAPI()

# read csv data
df = pd.read_csv("../data/processed/mastercard_stock_history_processed.csv", parse_dates = True, index_col = ["Date"])

# TODO: class of dataframe we want to send in request body
# class Data(BaseModel):

# forecast = ARIMAX.get_prediction(start = df1.index[-20], end = df1.index[-1], exog = df1[["Open", "Volume", "MA", "M_STD"]][-20:])
# mean_forecast = forecast.predicted_mean
# confidence_intervals = forecast.conf_int()

# load the models
ARIMAX = load("../models/arimax_model.pkl")
XGB = load("../models/best_XGBRegressor_v1.pkl")
CHRONOS_T5_SMALL = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")

# get requests to get predictions from the models
@app.get("/arimax")
def read_root():
    return {"Hello": "World"}