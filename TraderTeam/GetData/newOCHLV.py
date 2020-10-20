import pandas as pd
import requests


def make_csv():
    apikey = "EAGI0CX3I2LS6RDG"
    url = "https://www.alphavantage.co/query"
    for n in range(1, 13):
        params = {"function": "TIME_SERIES_INTRADAY_EXTENDED",
                    "symbol": "AAPL",
                    "interval": '1min',
                    "datatype": "csv",
                    "slice": f"year2month{n}",
                    "apikey": apikey}
        response = requests.get(url, params)
        with open(f"/Users/colincurtis/4castr/newData/AAPL_AV{12 + n}.csv", "wb") as f:
            f.write(response.content)


# make_csv()


def concatenate():
    df = pd.read_csv('/Users/colincurtis/4castr/newData/AAPL_AV1.csv')
    for n in range(2, 25):
        data = pd.read_csv(f"/Users/colincurtis/4castr/newData/AAPL_AV{n}.csv")
        df = df.append(data, ignore_index=True)
    df = df[::-1]
    df.reset_index(inplace=True, drop="index")
    return df


highFreqData = concatenate()

# highFreqData.set_index(range(len(highFreqData)), inplace=True)
# highFreqData = highFreqData.reindex(index=highFreqData.index[::-1])
print(highFreqData)

# highFreqData.to_csv("/Users/colincurtis/4castr/newData/AAPL_AV.csv")
