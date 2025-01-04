from typing import Tuple

from joblib import load

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMAResultsWrapper

class Arimax:

    def __init__(self, path):
        self.model = load(path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Preprocess data into the form that the saved ARIMAX model can use

        Args:
            df: pd.DataFrame. Dataframe of observations of past values

        Returns:
            df1: pd.DataFrame. DataFrame of processed data for ARIMAX model to use
        '''
        # apply log-transform so that variance is constant over time
        df["log_close"] = np.log(df["Close"])
        df["log_close_diff"] = np.log(df["Close"]).diff()
        df = df.dropna()

        scaler = StandardScaler()
        # the below are also regressors we will use to predict the price of the stock (log_close)
        cols_to_standardize = ["Open", "Volume", "MA", "M_STD"]
        df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

        cols_to_keep = ["Open", "Volume", "MA", "M_STD", "log_close", "Close"]
        df1 = df[cols_to_keep]

        return df1

    def predict(
        self,
        df_arima: pd.DataFrame, 
        days: int
    ) -> Tuple[pd.Series, pd.DataFrame]:
        '''
        Function to make predictions with the arimax model.

        Args:
            arimax_model: fitted arimax model that we will load from models folder
            df_arima: preprocessed dataframe for arimax
            days: number of days to predict, starting from the last timestamp arimax was trained on

        Returns:
            Tuple of predicted_prices, conf_int
        '''

        # forecasting stuff, according to number of days we want (NOTE: must start at index -20, as it is the last timestamp ARIMAX model was trained on)
        exog_slice_end = -(20 - days) if days < 20 else None
        forecast = self.model.get_prediction(start = df_arima.index[-20], end = df_arima.index[-(20 - days + 1)], exog = df_arima[["Open", "Volume", "MA", "M_STD"]][-20:exog_slice_end])
        mean_forecast = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()

        # raise it to the exponential power, since ARIMAX predicts log closing prices
        predicted_prices = np.exp(mean_forecast)
        conf_int = np.exp(confidence_intervals)

        # rename the keys in the dictionary
        conf_int["lower_close"] = conf_int.pop("lower log_close")
        conf_int["upper_close"] = conf_int.pop("upper log_close")

        return (predicted_prices, conf_int)
