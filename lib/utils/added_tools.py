import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time
from tensorflow.keras.callbacks import TensorBoard
import random
#...

def generate_actions():
  """
  Generates the array of tuples of actions, 
  since trader should have access to an amount to trade and and action to trade.
  """
  actions = []
  for i in range(3):
      for j in range(10):
        actions.append([i, j])
  return actions

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super(ModifiedTensorBoard, self).__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        names = names = ["loss", "acc"]
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = logs['loss']
        summary_value.tag = "loss"
        self.writer.add_summary(summary, self.step)

        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = logs['acc']
        summary_value.tag = "acc"
        self.writer.add_summary(summary, self.step)
        
    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, stats):
        i = 0
        names = ["average_reward", "min_reward", "max_reward", "epsilon"]
        for item in stats:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = item
            summary_value.tag = names[i]
            self.writer.add_summary(summary, self.step)
            i += 1

# These functions setup a static_environment
def dir_setup(mode):
  # UTIL FUNCTION --> creates directories
  
  maybe_make_dir('stories')
  maybe_make_dir('data')
  
  # Define a timestamp for currentl model
  timestamp = time.strftime('%Y%m%d%H%M')

  return timestamp


def live_env_setup(initial_invest, state_size, action_size):
  env = LiveEnv(initial_invest)
  return env, state_size, action_size 

