from TraderTeam.InputTensorOptimization.featureEngineering import taData
import numpy as np
import pandas as pd
from TraderTeam.InputTensorOptimization.corellationMatrix import plotVar


def apply_convolution(df, length):
    mean_kernel = [1/length] * length
    for column in df.columns:
        df[column] = np.convolve(df[column], mean_kernel, mode='same')
    return df


convolved = apply_convolution(taData, 100)

