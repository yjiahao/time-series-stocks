from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import pandas as pd

from utils.arimax import Arimax
from utils.xgb import XGB
from utils.chronos import Chronos

# class of data we want to send in post request
class PredData(BaseModel):
    days: int

class FastAPIApp:
    def __init__(self):
        # instantiate FastAPI app
        self.app = FastAPI()

        # read csv data
        self.df = pd.read_csv("../data/processed/mastercard_processed.csv", parse_dates = True, index_col = ["Date"])

        # load the models
        self.arimax = Arimax("../models/arimax_model.pkl")
        self.xgb = XGB("../models/best_XGBRegressor_v1.pkl")
        self.chronos = Chronos("amazon/chronos-t5-small")

        self.setup_routes()

    def setup_routes(self):
        # post endpoints to get predictions from the models
        @self.app.post("/arimax")
        def arima_preds(data: PredData):
            # preprocess data into the form that the ARIMAX model accepts
            df_arima = self.arimax.preprocess_data(self.df)
            # use arimax to predict the future prices and confidence intervals
            predicted_prices, conf_int = self.arimax.predict(df_arima, data.days)
            return {"preds": predicted_prices, "conf_int": conf_int}

        # post endpoint for XGB
        @self.app.post("/XGB")
        def xgb_preds(data: PredData):
            df_xgb = self.xgb.preprocess_data(self.df)
            # use XGB to predict the future prices
            predicted_prices = self.xgb.predict(df_xgb, data.days).tolist()
            return {"preds": predicted_prices}

        # post endpoint for the chronos model
        @self.app.post("/chronos")
        def chronos_preds(data: PredData):
            low, median, high = self.chronos.predict(self.df, data.days)
            low = low.tolist()
            median = median.tolist()
            high = high.tolist()
            return {"preds": median, "low": low, "high": high}

# need to expose the FastAPI instance for uvicorn
fastapi_app = FastAPIApp()
app = fastapi_app.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)