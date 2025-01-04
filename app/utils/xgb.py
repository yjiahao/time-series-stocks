from typing import Tuple

from joblib import load

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import xgboost

class XGB:
    def __init__(self, path):
        self.model = load(path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Preprocess data into the form that the saved XGB model can use

        Args:
            df: pd.DataFrame. Dataframe of observations of past values

        Returns:
            df: pd.DataFrame. DataFrame of processed data for XGB model to use
        '''
        # the only preprocessing step is to scale the data for XGB
        scaler = StandardScaler()
        cols_to_standardize = ['Open', "Volume", "lag_1", "lag_2", "MA", "M_STD"]
        df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
        return df

    def predict(
        self,
        df_xgb: pd.DataFrame,
        days: int
    ) -> np.ndarray:
        '''
        Function to make predictions with the XGB model.

        Args:
            xgb_model: fitted arimax model that we will load from models folder
            df_xgb: preprocessed dataframe for arimax
            days: number of days to predict

        Returns:
            predicted_prices: a numpy array of price predictions made by XGB
        '''
        preds = self.model.predict(df_xgb[["Open", "Volume", "month", "day", "quarter", "lag_1", "lag_2", "MA", "M_STD"]][-days:])
        return preds

# TODO: create another function that will predict and compute the stuff needed to predict prices in the future (not one step ahead predictions)