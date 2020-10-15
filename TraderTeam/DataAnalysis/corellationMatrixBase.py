import pandas as pd
import matplotlib.pyplot as plt

stockData = pd.read_csv('/data/AAPL_2018_2020_1hr.csv')
# print(stockData.head(15))
# print(stockData.columns)

# rollingStockData = stockData.rolling(5).mean().fillna(0)
# print(stockData.head(15))


def plotVar(column: str):  # plot a single variable, raw data
    plt.plot(stockData[column])
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.show()


def plotRollingVar(column: str):  # plot a single variable, raw data
    fig, ax = plt.subplots(2)
    fig.suptitle(f"Rolling vs Raw Data for {column}")
    ax[0].plot(stockData.rolling(3).mean().fillna(0)[column])
    ax[1].plot(stockData[column])
    plt.show()


# plotRollingVar('Close')

# def plotAllVars():
#     fig, ax = plt.subplots()
#     for column in stockData.columns:
#         ax.plot(stockData[column])
#     fig.show()


def plot_scatter_matrix():
    pd.plotting.scatter_matrix(stockData, diagonal="kde", alpha=.5, figsize=(12, 8))
    plt.suptitle("Scatter Matrix")
    # plt.rcParams.update({'font.size': 10})
    plt.show()


# plot_scatter_matrix()


def scatter(column1: str, column2: str):  # generate scatter plot between two columns of dataframe
    x = stockData[column1]
    y = stockData[column2]
    plt.scatter(x, y)
    correlation = x.corr(y)
    plt.title(f"Pearson correlation: {correlation}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.suptitle(f"Scatter plot between {column1} and {column2}")
    plt.show()


def rollingScatter(column1: str, column2: str):  # generate scatter plot between two columns of dataframe
    x = stockData.rolling(5).mean().fillna(0)[column1]
    y = stockData.rolling(5).mean().fillna(0)[column2]
    plt.scatter(x, y)
    correlation = x.corr(y)
    plt.title(f"Pearson correlation: {correlation}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.suptitle(f"Scatter plot between {column1} and {column2}")
    plt.show()


scatter("Open", "Adj Close")
rollingScatter("Open", "Adj Close")

# volume has terrible correlation with everything, might want to consider dumping it from tensor
# scatter("Open", "Volume")
# scatter("High", "Volume")
# scatter("Low", "Volume")
# scatter("Close", "Volume")
# scatter("Adj Close", "Volume")
