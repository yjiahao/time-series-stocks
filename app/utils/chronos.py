from typing import Tuple

import pandas as pd
import numpy as np

import torch

from chronos import ChronosPipeline

def chronos_predict(
    chronos_model: ChronosPipeline,
    df_chronos: pd.DataFrame,
    days: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Function to make predictions with the chronos model.

    Args:
        chronos_model: fitted arimax model that we will load from models folder
        df_chronos: preprocessed dataframe for arimax
        days: number of days to predict

    Returns:
        a tuple of np.ndarray, of low (10th percentile), median (50th percentile), and high (90th percentile)
    '''
    context = torch.tensor(df_chronos["Close"][:-days])
    forecast = chronos_model.predict(context, days)  # shape [num_series, num_samples, prediction_length]
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    return (low, median, high)

# TODO: explore the possibility of adding exogeneous variables to improve the predictions