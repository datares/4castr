import pandas as pd
import matplotlib.pyplot as plt

stockData = pd.read_csv('/Users/colincurtis/4castr/data/AAPL.csv')
stockData.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

rollingWindow = 30
dropIndex = list(range(rollingWindow - 1))

# droppedDF = stockData.rolling(rollingWindow)
# .mean().fillna(0).drop(index=dropIndex, columns='Volume', inplace=False).round(2)

momentum = stockData["High"].rolling(10).mean().dropna(axis=0)\
    .divide(stockData["High"].rolling(30).mean().dropna(axis=0))
momentum.columns = "Momentum"

meanVolume = stockData["Volume"].mean()  # roughly 4.5e8

intraDayChange = stockData["Open"] - stockData["Close"]
intraDayChange = intraDayChange.divide(stockData["Volume"].divide(meanVolume))
stockData.insert(len(stockData.columns), "Day Change", intraDayChange)


stockData.insert(len(stockData.columns), "Momentum", momentum)
stockData.dropna(axis=0, inplace=True)
stockData.round(3)
# print(stockData)
# stockData.to_csv("/Users/colincurtis/4castr/data/AAPL_newSignals.csv")


def plotVar(df, column: str):  # plot a single variable, raw data
    plt.plot(df[column])
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title(f"Plot for {column}")
    plt.show()


# plotVar("Day Change")


def plotRollingVar(df, column: str, window: int):  # plot a single variable, using a rolling window
    fig, ax = plt.subplots(2)
    fig.suptitle(f"{window} Day Rolling Avg vs Raw Data for {column}")
    ax[0].plot(df.rolling(window).mean().fillna(0)[column])
    ax[1].plot(df[column])
    plt.show()


def plot_scatter_matrix(df):
    pd.plotting.scatter_matrix(df, diagonal="kde", alpha=.5, figsize=(12, 8))
    plt.suptitle("Scatter Matrix")
    # plt.rcParams.update({'font.size': 10})
    plt.show()


def scatter(df, column1: str, column2: str):  # generate scatter plot between two columns of dataframe
    x = df[column1]
    y = df[column2]
    plt.scatter(x, y)
    correlation = x.corr(y)
    plt.title(f"Pearson correlation: {correlation}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.suptitle(f"Scatter plot between {column1} and {column2}")
    plt.show()


def rollingScatter(df, column1: str, column2: str):  # generate scatter plot between two columns of dataframe
    x = df.rolling(5).mean().fillna(0)[column1]
    y = df.rolling(5).mean().fillna(0)[column2]
    plt.scatter(x, y)
    correlation = x.corr(y)
    plt.title(f"Pearson correlation: {correlation}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.suptitle(f"Scatter plot between {column1} and {column2}")
    plt.show()

# volume has terrible correlation with everything, might want to consider dumping it from tensor


# rawCorrMatrix = stockData.corr()
# rollingCorrMatrix = stockData.rolling(rollingWindow).mean().fillna(0).corr()
#
# corrChange = (rollingCorrMatrix - rawCorrMatrix).divide(rawCorrMatrix)
# print(corrChange)

# Notice very little percent increase in corellation between OCHL raw and rolling values, but rather
# high percent change between the rest of the values

