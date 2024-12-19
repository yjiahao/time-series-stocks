import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def preprocess_arima_data(df: pd.DataFrame) -> pd.DataFrame:
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