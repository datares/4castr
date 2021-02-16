import gym
import numpy as np
import random
import joblib

from sklearn.preprocessing import MinMaxScaler
from lib.utils.added_tools import generate_actions, clamp


def get_scaler(mode, data):
    """
    Initializes or loads an sklearn MinMaxScaler, fits it to the data, and pickles it if training or optimizing. For more information on sklearn scalers, see the documentation.
    
    Args:
        mode (string): Whether training or optimizing, or testing.
    
    Returns:
        The generated/loaded MinMaxScaler.
    """
    scaler = None
    if mode == "train" or mode == "optimize":
        scaler = MinMaxScaler()
        scaler.fit(data)
        joblib.dump(scaler, 'saved_scaler.pkl')
        return scaler
    return joblib.load('saved_scaler.pkl')


class SimulatedEnv(gym.Env):
    """
    Simulated environment for trading stocks. Extends gym.Env.
    """
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self, data, init_invest=1000, mode="test"):
        """
        Initializes a SimulatedEnv.
        
        Args:
            data (pandas.DataFrame): DataFrame consisting of OHLC data of the user's choice.
            init_invest (int): Starting budget. Defaults to 1000.
            mode (string): Whether we are training, finetuning, or testing the model. Defaults to 'test'.
        
        Returns:
            None
        """
        super(SimulatedEnv, self).__init__()
        self.dataset = data
        self.mode = mode
        self.n_steps = 180
        self.finish_step = 0
        self.init_invest = init_invest
        self.cur_step = None
        self.current_state = []
        self.owned_stocks = None
        self.cash_in_hand = None
        self.scaler = get_scaler(mode, self.dataset)

        self.actions = generate_actions()
        self.action_space = gym.spaces.Discrete(30)

        self.n_features = data.shape[1]
        self.obs_shape = (1, self.n_features)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=self.obs_shape,
                                                dtype=np.float16)
        self.reset()

    # def _seed(self, seed=None):
    #   self.np_random, seed = seeding.np_random(seed)
    #   return [seed]

    def reset(self):
        """
        Resets the environment. 
        
        Args:
            None
        
        Returns:
            Current state as a list.
            
        """
        limit = len(self.dataset) - self.n_steps
        self.cur_step = random.randrange(limit)
        self.finish_step = self.cur_step + self.n_steps
        self.owned_stocks = 0
        self.cash_in_hand = self.init_invest
        self.current_state = self.dataset.iloc[self.cur_step, :]

        return self._get_obs()

    def step(self, action):
        """
        Takes a step by performing an action.
        
        Args:
            action (int): Action to be taken.
        
        Returns:
            Tuple containing the new state, reward after taking the action, bool describing whether the agent is done taking steps, and a dict containing action information.
        """
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        self.current_state = self.dataset.iloc[self.cur_step, :]
        open_price = self.current_state[0]

        # Trade with an action
        self._trade(action)
        # Check on how much it changed
        cur_val = self._get_val()

        reward = clamp(-5, cur_val - prev_val, 5)
        # reward = cur_val - prev_val

        done = (self.cur_step >= self.finish_step - 1)
        info = {'cur_val': cur_val,
                'prev_val': prev_val,
                'action': action,
                'owned_stocks': self.owned_stocks,
                'cash_in_hand': self.cash_in_hand,
                'price': open_price
                }

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """
        Gets the current state and scales it.
        
        Args:
            None.
        
        Returns:
            List containing the scaled state.
        """
        obs = []
        for element in self.current_state:
            obs.append(element)

        obs = np.array(obs)
        return self.scaler.transform([obs])

    def _get_val(self):
        """
        Gets the agent's net worth.
        
        Args:
            None.
        
        Returns:
            (Float) agent's net worth.
        """
        open_price = self.current_state[0]
        return np.sum(self.owned_stocks * open_price) + self.cash_in_hand

    def _trade(self, action):
        """
        Buys or sells a stock and updates the buying power of the agent accordingly.
        
        Args:
            action (int): Action taken (buy, sell, or hold).
        
        Returns:
            None
        """
        combo = self.actions[action]
        move = combo[0]
        amount = combo[1]
        open_price = self.current_state[0]
        expense = amount * open_price

        sell = False
        buy = False

        if move == 0:
            sell = True
        elif move == 2:
            buy = True

        if sell and self.owned_stocks >= amount:
            self.cash_in_hand += expense
            self.owned_stocks -= amount

        if buy and self.cash_in_hand >= expense:
            self.cash_in_hand -= expense
            self.owned_stocks += amount

    def render(self, mode='system'):
        """
        Prints the price of a stock and the net worth of the agent.
        
        Args:
            mode (string) : #TODO
        
        Returns:
            List containing the scaled state.
        """
        if mode == 'system':
            print('Price: ' + str(self.current_state[0]))
            print('Net worth: ' + str(self._get_val))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
