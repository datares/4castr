# UTILITY LIBRARIES
import numpy as np
import pandas as pd
import joblib
import datetime
import coloredlogs
from yahoo_fin import stock_info as si
from stable_baselines import PPO2

# UTILS
from lib.utils.added_tools import dir_setup, generate_actions
from lib.utils.generate_ta import create_ta, clean_ta
from lib.utils.logger import init_logger

# DEFINE GLOB VARIABLES
ACTIONS = ["SELL", "HOLD", "BUY"]


def real_time_yahoo(stock):
    end_date = pd.Timestamp.today() + pd.DateOffset(10)
    data = si.get_data(stock, end_date=end_date)
    data = create_ta(data)
    data = data.fillna(0)
    data = clean_ta(data)
    data = data.iloc[-1:, :]
    return data


class Live_Session:
    def __init__(self, mode, initial_invest, session_name, brain, stock):
        self.session_name = session_name
        self.mode = mode
        self.initial_invest = initial_invest
        self.portfolio_value = []
        self.test_ep_rewards = []
        self.actions = generate_actions()
        self.timestamp = dir_setup(mode)
        self.env = None
        self.scaler = joblib.load('saved_scaler.pkl')
        self.stock = stock
        self.logger = init_logger(__name__, show_debug=True)
        self.brain = PPO2.load(brain)
        coloredlogs.install(level='TEST', logger=self.logger)
        self.logger.info("Bot is live: [{}]".format(datetime.datetime.now()))

    def _get_obs(self):
        state = real_time_yahoo(self.stock).values
        self.logger.info("Received state as: {}".format(state[0]))
        obs = []
        for element in state:
            obs.append(element)
        return np.array(obs)

    def get_action(self):
        obs = self.scaler.transform(self._get_obs())
        action, _states = self.brain.predict(obs)
        combo = self.actions[action]
        move = combo[0]
        amount = combo[1]
        return ACTIONS[move], amount

    def go_live(self, s_repeat=3600, steps=180):
        import time
        for i in range(steps):
            d = datetime.datetime.now()
            a = self.get_action()
            self.logger.info(f"[{d}] \nModel Action: {a}")
            time.sleep(s_repeat)
