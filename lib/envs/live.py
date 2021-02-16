import gym
from gym import spaces
from gym.utils import seeding

import time
import numpy as np
import itertools
import random
import pandas as pd
from ta import *
import requests
import json

OHLC_URL = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/latest?period_id=1MIN'
COINAPI_KEY = {'X-CoinAPI-Key': '73034021-0EBC-493D-8A00-E0F138111F41'}


def get_live_OHLCV():
    """
    Gets the latest OHLC data for bitcoin, stores it in a dict, and returns it.

    Args:
        None

    Returns:
        Dict containing latest BTC OHLC data.
    """
    d = {'price_open': [], 'price_low': [],  'price_high': [],
         'price_close': [], 'volume_traded': []}

    response = requests.get(OHLC_URL, headers=COINAPI_KEY)
    latest_json_data = json.loads(response.text)

    for i in range(0, len(latest_json_data)):
        d['price_open'].append(latest_json_data[i]['price_open'])
        d['price_high'].append(latest_json_data[i]['price_high'])
        d['price_low'].append(latest_json_data[i]['price_low'])
        d['price_close'].append(latest_json_data[i]['price_close'])
        d['volume_traded'].append(latest_json_data[i]['volume_traded'])
    latest_json_data = pd.DataFrame(data=d)

    return latest_json_data


class LiveEnv():
    """
    Live environment for trading stocks.
    """

    def __init__(self, init_invest=1000, action_space=30, state_space=63):
        """
        Initializes a SimulatedEnv.

        Args:
            init_invest (int): Starting budget. Defaults to 1000.
            action_space (int): Size of the action space. Defaults to 30.
            state_space (int): Size of the state space. Defaults to 63.

        Returns:
            None
        """
        # Amount of money we are initially putting in.
        self.init_invest = init_invest

        # How much crypto you have.
        self.crypto_owned = None

        self.n_step = 1000

        # How cash you have.
        self.cash_in_hand = None

        # Action space
        self.action_space = spaces.Discrete(action_space)

        self._reset()

    def _reset(self):
        """
        Resets the environment.

        Args:
            None

        Returns:
            List containing current state.
        """
        # Re initialize crypto owned, cash in the hand, and current state.
        self.crypto_owned = 0
        self.cash_in_hand = self.init_invest

        return self._get_obs()

    def _step(self, action):
        """
        Takes a step by performing an action.

        Args:
            action (int): Action to be taken.

        Returns:
            Tuple containing the new state, reward after taking the action, and a dict containing action information.
        """
        assert self.action_space.contains(action)
        # Get OHLCV for crypto. You want the most recent and column 0 is the open price.
        self.open_price = get_live_OHLCV().iloc[0, 0]
        # Get previous value of portfolio
        prev_val = self._get_val()

        # Trade with an action
        self._trade(action)

        # Check on how much it changed
        cur_val = self._get_val()

        # How much did we grow from the previous step?
        reward = cur_val - prev_val

        info = {'cur_val': cur_val, 'prev_val': prev_val, 'action': action,
                'crypto_owned': self.crypto_owned, 'cash_in_hand': self.cash_in_hand, 'price': self.open_price}

        time.sleep(60)

        return self._get_obs(), reward, info

    def _get_obs(self):
        """
        Gets the current state.

        Args:
            None.

        Returns:
            List containing the state.
        """
        obs = []
        data = get_live_OHLCV()
        live_ta = add_all_ta_features(
            data, "price_open", "price_high", "price_low", "price_close", "volume_traded", fillna=True)
        live_ta = live_ta.iloc[0, :]
        for element in live_ta:
            obs.append(element)
        return obs

    def _get_val(self):
        """
        Gets the agent's net worth.

        Args:
            None.

        Returns:
            (Float) agent's net worth.
        """
        return np.sum(self.crypto_owned * self.open_price) + self.cash_in_hand

    def _trade(self, action):
        """
        Buys or sells a stock and updates the buying power of the agent 
        accordingly.

        Args:
            action (int): Action taken (buy, sell, or hold).

        Returns:
            None
        """
        # Compute price of crypto in dollars and a dollar of crypto
        price_dollar = self.open_price
        price_crypto = self.cash_in_hand / price_dollar
        sell = False
        buy = False

        if action == 0:
            sell = True
        elif action == 2:
            buy = True

        if sell and self.crypto_owned is not 0:
            self.cash_in_hand = self.crypto_owned * price_dollar
            self.crypto_owned = 0

        if buy and self.cash_in_hand is not 0:
            self.crypto_owned = price_crypto
            self.cash_in_hand = 0
