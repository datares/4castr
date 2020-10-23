# UTILITY LIBRARIES
import numpy as np
import pandas as pd
import optuna
from yahoo_fin import stock_info as si
import coloredlogs
from sklearn.model_selection import train_test_split

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
MINI_DATA_PATH = "./newData/AAPL_normalized.csv"  # rolling csv dataframe.


def manual_agent_params(SOMETHING: int):
    return {
        'n_steps': 10000,
        'gamma': 0.99,
        'learning_rate': .00010179263199758284,  # was .00010179263199758284
        'ent_coef': 0.01,
        'cliprange': 0.2668120684510983,
        'noptepochs': 10,  # changed from 5 to 30
        'lam': 0.95
    }


def historical_yahoo(stock):
    data = si.get_data(stock,
                       end_date=pd.Timestamp.today() + pd.DateOffset(10))
    data = create_ta(data)
    data = data.fillna(0)
    data = clean_ta(data)
    return data


def train_val_test_split(path):
    data = pd.read_csv(path)
    # train_data, test_data = train_test_split(data, train_size=.9, test_size=.1, shuffle=False)
    train_data = data.iloc[:-10000, 1:]
    test_data = data.iloc[-10000:-2500, 1:]
    validation_data = data.iloc[-2500:, 1:]
    return train_data, validation_data,  test_data


def train_val_test_split_finetune(data):
    train_data = data.iloc[:-10000, :]
    test_data = data.iloc[-10000:, :]
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
    def __init__(self,
                 mode,
                 test_episodes,
                 initial_invest,
                 session_name,
                 stock=None,
                 brain=None):
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
        self.n_steps_per_eval = 100000000
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
        # Print stats
        self.logger.info("episode: {}/{}, episode end value: {}".format(
          e + 1, self.test_episodes, info[0]['cur_val']))
        loss_percent = int(self.losses / (e + 1) * 100)
        self.logger.info(f"loss percent --> {loss_percent}%")

    def write_to_story(self, f, action, info):
        f.write("{},{},{},{},{},{}\n".format(ACTIONS[self.actions[action][0]],
                                             self.actions[action][1],
                                             info['owned_stocks'],
                                             info['cash_in_hand'],
                                             info['cur_val'],
                                             info['price']))
    # OPTIMIZE FUNCTIONS

    def optimize_agent_params(self, trial):
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
