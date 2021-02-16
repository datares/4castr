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
    """
    Generates and formats a pandas DataFrame containing OHLC data of a 
    chosen stock up through the current day.
    
    Args:
        stock (string): Ticker representing a stock.
    
    Returns:
        Pandas dataframe containing current OHLC data. 
    """
    end_date = pd.Timestamp.today() + pd.DateOffset(10)
    data = si.get_data(stock, end_date=end_date)
    data = create_ta(data)
    data = data.fillna(0)
    data = clean_ta(data)
    data = data.iloc[-1:, :]
    return data


class Live_Session:
    """
    Live session of stock trading.
    """
    def __init__(self, mode, initial_invest, session_name, brain, stock):
        """
        Initializes a Live_Session.
        
        Args:
            mode (string): Whether training, finetuning, or testing.
            initial_invest (int): Starting budget.
            brain (stable_baselines.ppo2.PPO2 model): Model to use.
            stock (string): Ticker representing a chosen stock.
        
        Returns:
            None
        """
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
        """
        Finds the current state of its Live_Session object's stock.
        
        Args:
            None
            
        Returns:
            Array of observations representing current stock data.
        """
        state = real_time_yahoo(self.stock).values
        self.logger.info("Received state as: {}".format(state[0]))
        obs = []
        for element in state:
            obs.append(element)
        return np.array(obs)

    def get_action(self):
        """
        Takes an action given the state as a prediction by a PPO2 model. 
        
        Args:
            None
            
        Returns:
            Tuple containing the action (buy, sell, or hold) and the amount.
        """
        obs = self.scaler.transform(self._get_obs())
        action, _states = self.brain.predict(obs)
        combo = self.actions[action]
        move = combo[0]
        amount = combo[1]
        return ACTIONS[move], amount

    def go_live(self, s_repeat=3600, steps=180):
        """
        Buys and sells stocks in real time at interval 's_repeat', for 'steps' steps.
        
        Args:
            s_repeat (int): Time between actions taken in real time. Defaults to 3600.
            steps (int): Number of steps to run for. Defaults to 180.
            
        Returns:
            Array of observations representing current stock data.
        """
        import time
        for i in range(steps):
            d = datetime.datetime.now()
            a = self.get_action()
            self.logger.info(f"[{d}] \nModel Action: {a}")
            time.sleep(s_repeat)
