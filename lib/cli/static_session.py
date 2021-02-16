# UTILITY LIBRARIES
import numpy as np
import pandas as pd
import optuna
from yahoo_fin import stock_info as si
import coloredlogs

from lib.utils.logger import init_logger
from lib.utils.generate_ta import create_ta, clean_ta

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

# ENVIRONMENT
from lib.envs.simulated import SimulatedEnv
# UTILS
from lib.utils.added_tools import dir_setup, generate_actions
from config import TOTAL_DATA_PATH

# GLOB VARIABLES
ACTIONS = ["SELL", "HOLD", "BUY"]
MINI_DATA_PATH = "./data/AAPL_2018_2020_1hr.csv"


def manual_agent_params(SOMETHING: int):
    """
    Directly returns optimal hyperparameters for PPO2 model.
    
    Args:
        SOMETHING (int): Unused.
        
    Returns:
        Dictionary containing hyperparameters.
    """
    return {
        'n_steps': 1024,
        'gamma': 0.9391973108460121,
        'learning_rate': 0.00010179263199758284,
        'ent_coef': 0.0001123894292050861,
        'cliprange': 0.2668120684510983,
        'noptepochs': 5,
        'lam': 0.8789545362092943
    }


def historical_yahoo(stock):
    """
    Generates a pandas DataFrame from OHLC data from Yahoo finance.
    
    Args:
        stock (string): Ticker representing a stock.
    
    Returns:
        Pandas dataframe of OHLC history of chosen stock.
    """
    data = si.get_data(stock,
                       end_date=pd.Timestamp.today() + pd.DateOffset(10))
    data = create_ta(data)
    data = data.fillna(0)
    data = clean_ta(data)
    return data


def train_val_test_split(path):
    """
    Splits a dataset into train, validation and test sets.
    
    Args:
        path (String): Path to the data
    
    Returns:
        train_data, validation_data, test_data (tuple): Pandas DataFrames 
        containing split data
    """
    data = pd.read_csv(path)
    train_data = data.iloc[:-400, 1:]
    validation_data = data.iloc[-400:-50, 1:]
    test_data = data.iloc[-50:, 1:]
    return train_data, validation_data, test_data


def train_val_test_split_finetune(data):
    train_data = data.iloc[:-2000, :]
    test_data = data.iloc[-2000:, :]
    return train_data, test_data


def make_env(env, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you
                          wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


class Static_Session:
    """
    Static session of stock trading. Used to train agents before going live.
    """
    def __init__(self,
                 mode,
                 test_episodes,
                 initial_invest,
                 session_name,
                 stock=None,
                 brain=None):
        """
        Initializes a Static_Session.
        
        Args:
            mode (string): Whether training, finetuning, or testing.
            test_episodes (int): How many episodes to test the model for.
            initial_invest (int): Starting budget.
            stock (string): Ticker representing a chosen stock. Defaults to None.
            brain (stable_baselines model): Model to use. Defaults to None.
        
        Returns:
            None
        """
        # SESSION_VARIABLES
        self.session_name = session_name
        self.mode = mode
        self.test_episodes = test_episodes
        self.initial_invest = initial_invest
        self.portfolio_value = []
        self.test_ep_rewards = []
        self.losses = 0
        self.actions = generate_actions()
        self.timestamp = dir_setup(mode)
        self.env = None
        self.brain = brain
        self.n_steps_per_eval = 10000000
        self.logger = init_logger(__name__, show_debug=True)
        self.stock = stock
        coloredlogs.install(level='TEST')

        self.train_ds, self.val_ds, self.test_ds = \
            train_val_test_split(TOTAL_DATA_PATH)

        self.optuna_study = optuna.create_study(
              study_name="{}_study".format(self.session_name),
              storage="sqlite:///data/params.db",
              load_if_exists=True,
              pruner=optuna.pruners.MedianPruner())

        self.logger.debug(f'Initialized Static Session: {self.session_name}')
        self.logger.debug(f'Mode: {self.mode}')

    # OUTPUT FUNCTIONS

    def print_stats(self, e, info):
        """
        Prints loss and episode by sending it to a logger object.
        
        Args:
            e (int): Episode
            info (dict): Model performance metrics returned by 
                stable_baselines env.step()
        
        Returns:
            None
        """
                
        # Print stats
        self.logger.info("episode: {}/{}, episode end value: {}".format(
          e + 1, self.test_episodes, info[0]['cur_val']))
        loss_percent = int(self.losses / (e + 1) * 100)
        self.logger.info(f"loss percent --> {loss_percent}%")

    def write_to_story(self, f, action, info):
        """
        Writes the status of the agent to a file.
        
        Args:
            f (string): Filename
            action (int): Action taken, corresponding to one of the three specified in global ACTIONS
            info (dict): Model performance metrics returned by stable_baselines env.step()
        
        Returns:
            None
        """
        f.write("{},{},{},{},{},{}\n".format(ACTIONS[self.actions[action][0]],
                                             self.actions[action][1],
                                             info['owned_stocks'],
                                             info['cash_in_hand'],
                                             info['cur_val'],
                                             info['price']))
    # OPTIMIZE FUNCTIONS

    def optimize_agent_params(self, trial):
        """
        Generates a set of hyperparameters by sampling randomly from a given range for each hyperparameters.
        
        Args:
            trial (optuna.trial): Trial object to generate hyperparameter values.
        
        Returns:
            Dictionary of sampled hyperparameters.
        """
        return {
            'n_steps': int(trial.suggest_loguniform('n_steps', 512, 2048)),
            'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate',
                                                      1e-5, 1.),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
            'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
            'lam': trial.suggest_uniform('lam', 0.8, 1.)
        }

    def optimize_params(self,
                        trial,
                        n_prune_evals_per_trial: int = 2,
                        n_tests_per_eval: int = 1):
        model_params = self.optimize_agent_params(trial)
        train_env = DummyVecEnv([lambda: SimulatedEnv(self.train_ds,
                                                      self.initial_invest,
                                                      self.mode)])
        model = PPO2(MlpPolicy,
                     train_env,
                     verbose=0,
                     tensorboard_log="./logs/",
                     nminibatches=1,
                     **model_params)
        
        """
        Calculates reward given a trial with hyperparameters generated by optimize_agent_params.
        
        Args:
            trial (optuna.trial): Trial object to generate hyperparameter
                values.
            n_prune_evals_per_trial (int): Number of times to evaluate 
                the model and prune if necessary. Defaults to 2.
            n_tests_per_eval (int): Unused.
            model_params (dict): Dictionary of hyperparameters generated by 
                optimize_agent_params. Defaults to self.optimize_agent_params(trial).
            train_env (stable_baselines.common.vec_env.DummyVecEnv): training 
                environment on which PPO2 is trained. Defaults to a SimulatedEnv.
            model(stable_baselines model): Model to optimize parameters for. 
                Defaults to PPO2.
            
        Returns:
            (float) Model reward.
        """
        
        # TODO fix this
        
        for eval_idx in range(n_prune_evals_per_trial):
            try:
                model.learn(self.n_steps_per_eval)
            except AssertionError:
                raise

            last_reward = self.run_test(model)

            trial.report(-1 * last_reward, eval_idx)
            if trial.should_prune(eval_idx):
                raise optuna.structs.TrialPruned()
        return -1 * last_reward

    def get_model_params(self):
        """
        Loads optimized hyperparameters.
        
        Args:
            None
            
        Returns:
            Dict containing optimized hyperparameters.
        """
        params = self.optuna_study.best_trial.params
        self.logger.debug('Loaded best parameters as: {}'.format(params))

        return {
            'n_steps': int(params['n_steps']),
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['cliprange'],
            'noptepochs': int(params['noptepochs']),
            'lam': params['lam'],
        }

    def run_optimization(self, n_trials: int = 10):
        """
        Runs a hyperparameter search and selects the best based on highest 
        reward.
        
        Args:
            n_trials (int): number of trials to run. Each a unique set of 
                hyperparameters. Defaults to 10.
            
        Returns:
            Dataframe of trial results.
        """
        try:
            self.optuna_study.optimize(self.optimize_params,
                                       n_trials=n_trials,
                                       n_jobs=1)
        except KeyboardInterrupt:
            pass

        finished_trials = len(self.optuna_study.trials)
        best_trial = self.optuna_study.best_trial.value

        self.logger.info(f'Finished trials: {finished_trials}')
        self.logger.info(f'Best trial: {best_trial}')

        return self.optuna_study.trials_dataframe()

    # TRAINING AND TESTING
    def run_test(self, model,
                 validation=True, finetune=False,
                 out_file=False, verbose=True):
        """
        Validates a trained model.
        
        Args:
            model (stable_baselines model): Model to be tested.
            validation(bool): Whether or not the model is to be validated on 
                validation data. Defaults to True.
            finetune(bool): Whether or not the model is to be tested on 
                external dataset. Defaults to False.
            out_file(bool): Whether or not to write model stats to output file.
                Defaults to False.
            verbose(bool): Whether or not to print model stats.
                Defaults to True.
            
        Returns:
            Mean of the total reward of the model.
        """
        f = None
        if out_file:
            f = open(f"stories/{self.timestamp}-{self.mode}-BTC.csv", "w+")
            f.write()
        env = None

        if not finetune:
            if validation:
                env = DummyVecEnv([lambda: SimulatedEnv(self.val_ds,
                                                        self.initial_invest,
                                                        self.mode)])
            else:
                env = DummyVecEnv([lambda: SimulatedEnv(self.test_ds,
                                                        self.initial_invest,
                                                        self.mode)])
        else:
            data = historical_yahoo("NKE")
            self.logger.debug('Downloaded Data from yahoo finance.')

            _, test_data = train_val_test_split_finetune(data)
            env = DummyVecEnv([lambda: SimulatedEnv(test_data,
                                                    self.initial_invest,
                                                    self.mode)])
            self.logger.debug('Downloaded Data from yahoo finance.')

        total_reward = []

        for e in range(self.test_episodes):
            # Reset the environment at every episode.
            state = env.reset()
            # Initialize variable to get reward stats.
            for _ in range(0, 180):
                action, _states = model.predict(state)
                next_state, reward, done, info = env.step(action)

                if out_file:
                    self.write_to_story(f, action[0], info[0])

                total_reward.append(reward)
                state = next_state

                if done:
                    if info[0]['cur_val'] < self.initial_invest:
                        self.losses = self.losses + 1
                    if verbose:
                        self.print_stats(e, info)
                    if out_file:
                        f.write("-1,-1,-1,-1,-1,-1\n")
                    break
        self.losses = 0
        return np.mean(total_reward)

    def run_train(self):
        """
        Trains a PPO2 algorithm on stock trading environment SimulatedEnv. 
        Saves the model via pickle when finished.
        
        Args:
            None
            
        Returns:
            None
        """
        train_env = DummyVecEnv([lambda: SimulatedEnv(self.train_ds,
                                                      self.initial_invest,
                                                      self.mode)])

        # model_params = self.get_model_params()
        model_params = manual_agent_params()
        model = PPO2(MlpPolicy,
                     train_env,
                     verbose=1,
                     nminibatches=1,
                     tensorboard_log="./logs/",
                     **model_params)
        try:
            model.learn(total_timesteps=self.n_steps_per_eval)
            result = self.run_test(model, validation=False)

            self.logger.info("test_mean --> {}".format(result))
            model.save("{}.pkl".format(self.session_name))

        except KeyboardInterrupt:
            print("Saving model...")
            model.save("{}.pkl".format(self.session_name))

    def fine_tune(self, stock, model_path, ts=1000000):
        """
        Finetunes a chosen model on a new/external stock OHLC dataset.
        
        Args:
            stock (string): Ticker specifying which stock to pull data from.
            model_path (string): Path to trained model to finetune.
            ts (int): Number of timesteps to train for.
            
        Returns:
            None
        """
        assert self.brain is not None

        data = historical_yahoo(stock)
        train_data, _ = train_val_test_split_finetune(data)

        fine_tune_env = SimulatedEnv(train_data,
                                     self.initial_invest,
                                     self.mode)
        fine_tune_env = DummyVecEnv([lambda: fine_tune_env])

        model = PPO2.load(model_path)
        model.set_env(fine_tune_env)

        self.logger.info("Finetuning for {}...".format(stock))

        model.learn(total_timesteps=ts)
        model.save("{}__{}.pkl".format(self.session_name, stock))
        self.logger.info("Saved as {}__{}.pkl".format(self.session_name,
                                                      stock))
        self.run_test(model, finetune=True)
