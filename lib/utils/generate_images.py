import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from added_tools import maybe_make_dir
import argparse

parser = argparse.ArgumentParser()
# Required parameters 
parser.add_argument('-f', '--file', type=str, required=True,
                    help='file to generate')

args = parser.parse_args()

maybe_make_dir("./{}/{}".format("images", args.file[:-4]))

data = pd.read_csv(args.file)

action_value = data.iloc[:, 0].values
portfolio_value = data.iloc[:, 4].values
stock_value = data.iloc[:, 5].values

episodes = []
episodes_actions = []
episodes_stocks = []
count = 0
prev = 0

for i in range(len(portfolio_value)):
    if portfolio_value[i] == -1:
        episodes.append(portfolio_value[prev:i])
        episodes_actions.append(action_value[prev:i])
        episodes_stocks.append(stock_value[prev:i])
        prev = i
        count = count + 1
with plt.style.context('Solarize_Light2'):
    for i in range(0, len(episodes)):
        fig, ax = plt.subplots()
        fig.set_figheight(9)
        fig.set_figwidth(16)
        ep_num = i
        plt.subplot(2, 1, 1)
        plt.grid()
        plt.plot(episodes[ep_num][1:])

        for i in range(len(episodes[ep_num])):
            if episodes_actions[ep_num][i] == "BUY":
                plt.scatter(i, episodes[ep_num][i], color="yellow", marker=".", label='BUY')
            if episodes_actions[ep_num][i] == "SELL":
                plt.scatter(i, episodes[ep_num][i], color="green", marker=".", label='SELL')
            if episodes_actions[ep_num][i] == "HOLD":
                plt.scatter(i, episodes[ep_num][i], color="grey", marker=".", label='HOLD')

        plt.subplot(2, 1, 2)
        plt.grid()
        plt.plot(episodes_stocks[ep_num][1:])
        
        plt.savefig("./{}/{}/{}.png".format("images",args.file[:-4], ep_num), dpi=300)



