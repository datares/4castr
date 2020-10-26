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
                  "slice": f"year=1month{n}",
                  "apikey": apikey}
        response = requests.get(url, params)
        with open(f"/Users/colincurtis/4castr/newData/AAPL_AV{n}.csv", "wb") as f:
            f.write(response.content)
    for n in range(13, 25):
        params = {"function": "TIME_SERIES_INTRADAY_EXTENDED",
                  "symbol": "AAPL",
                  "interval": '1min',
                  "datatype": "csv",
                  "slice": f"year2month{n}",
                  "apikey": apikey}
        response = requests.get(url, params)
        with open(f"/Users/colincurtis/4castr/newData/AAPL_AV{n}.csv", "wb") as f:
            f.write(response.content)


def concatenate(num_files: int):  # take range of csv files to create dataframe
    if num_files > 24 or num_files < 2:
        raise Exception("You can only input months in the range 2-24")
    else:
        df = pd.read_csv('/Users/colincurtis/4castr/newData/AAPL_AV1.csv')
        for n in range(2, num_files + 1):
            data = pd.read_csv(f"/Users/colincurtis/4castr/newData/AAPL_AV{n}.csv")
            df = df.append(data, ignore_index=True)
        df = df[::-1]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df.reset_index(inplace=True, drop=["index", "Date"])
        return df


highFreqData = concatenate(25)


# highFreqData.to_csv("/Users/colincurtis/4castr/newData/AAPL_AV.csv")
