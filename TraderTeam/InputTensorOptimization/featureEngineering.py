import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt

from TraderTeam.GetData.newOCHLV import highFreqData
from TraderTeam.InputTensorOptimization.corellationMatrix import plot_scatter_matrix


def makeFilter(df):
    rsi = ta.momentum.rsi(df["Close"], n=8, fillna=False).fillna(0)
    # relative strength index

    aroon = ta.trend.AroonIndicator(df["Close"], n=25, fillna=True)  # instantiate class

    aroon_indicator = aroon.aroon_indicator()

    moving_average_con_div = ta.trend.MACD(df["Close"], n_fast=26, n_slow=9, n_sign=9, fillna=True)
    macd = moving_average_con_div.macd()

    awesome = ta.momentum.ao(high=df["High"], low=df["Low"], fillna=True)

    df.insert(len(df.columns), "RSI", rsi)
    df.insert(len(df.columns), "AO", awesome)
    df.insert(len(df.columns), "Aroon Ind", aroon_indicator)
    df.insert(len(df.columns), "MACD", macd)

    df = df.round(4)
    return df


def z_score(df, window):  # normalize our data to workable numbers using z-score.  Roughly range -3-3
    r = df.rolling(window=window)
    m = r.mean()
    s = r.std()
    z = (df - m) / s
    z = z.dropna()
    z.reset_index(inplace=True, drop='index')
    return z


# taData = makeFilter(highFreqData)
# taData.drop(columns=["Date"], inplace=True)
# taData.drop(columns=["Date", "Open", "High", "Low", "Close", "Volume"], inplace=True)
# taData = z_score(taData, 200)
# taData.to_csv("/Users/colincurtis/4castr/newData/AAPL_normalized.csv")


def momentum(df):  # TODO: fix error: series are mutable objects and thus they cannot be hashed.
    momentum_short = df["Close"].rolling(10).mean().dropna(axis=0)
    momentum_long = df["High"].rolling(30).mean().dropna(axis=0)
    momentum_ind = momentum_short.divide(momentum_long)
    df.insert(len(df.columns), momentum_ind, 'Momentum')
    return df


def intraday_change(df):  #
    mean_volume = df['Volume'].mean()
    intra_day_change = df["Open"] - df["Close"]
    intra_day_change = intra_day_change.divide(df["Volume"].divide(mean_volume))
    df.insert(len(df.columns), "Day Change", intra_day_change)
    return df


def fourier_plot(df, column: str):
    array = df[column]
    sample_spacing = 1.0 / 800.0
    sample_pts = 1000
    xf = np.linspace(0.0, 1.0 / (2.0 * sample_spacing), sample_pts // 2)
    yf = fft(array)
    plt.title(f"fourier transform for {column}")
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.plot(xf, 2.0/sample_pts * np.abs(yf[0:sample_pts//2]))
    plt.show()


def fourierPlotAllCols(df):
    for column in df.columns:
        fourier_plot(df, column)


def histPlot(df):
    for column in df.columns:
        plt.title(f"histogram for {column}")
        plt.hist(df[column], bins=100)
        plt.show()


