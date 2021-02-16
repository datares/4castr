from collections import deque
import random
import numpy as np
from lib.agents.models.mlp import mlp
import time
from tensorflow.keras.callbacks import TensorBoard
from lib.utils.added_tools import ModifiedTensorBoard
import tensorflow as tf


class DDQNAgent(object):
    """ A simple Deep Q agent """

    def __init__(self, state_size, action_size, mode):
        """
        Initializes a Double DQN agent.

        Args:
            state_size (int): Size of the state space.
            action_size (int): Size of the action vector.
            mode (string): Model purpose 
                (training, finetuning, validation, testing).

        Returns:
            None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9975
        self.tau = 0.1
        self.mode = mode
        self.online_weights = 0
        self.target_weights = 0

        self.online_net = mlp(state_size, action_size)
        self.target_net = mlp(state_size, action_size)

        self.target_net.set_weights(self.online_net.get_weights())

        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}".format(time.time()))

    def remember(self, state, action, reward, next_state, done):
        """
        Appends metrics to memory.

        Args:
            state (int): int representing state to act in.
            action (int): int representing action taken.
            reward (int): int representing reward 
              after taking the action in the given state.
            next_state (int): int representing the state reached
              after taking an action in the previous state.
            done (bool): Whether or not agent has reached maximum steps.

        Returns:
            None
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Takes an action, sometimes randomly.

        Args:
            state (int): int representing state to act in.

        Returns:
            Action taken, represented as an int.
        """
        # Do something randomly
        if np.random.rand() <= self.epsilon and self.mode == "train":
            return random.randrange(self.action_size)
        act_values = self.online_net.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=32):
        """
        Experience replay. Stores batch_size random samples from target
        network, freezes target network weights while training online network
        on the batch. When done, updates target net weights. This approach
        improves stability of DQN as opposed to updating both
        simultaneously by allowing the online network to approximate a
        target network that does not change for the duration of training.

        Args:
            batch_size (int): Number of samples to take from memory.

        Returns:
            None
        """
        # implement vectors -> take a minibatch from the tuple and treat
        #  q values in the same way.
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])
        # Q(s)
        target = self.online_net.predict(states)
        # Q(s')
        target_next = self.online_net.predict(next_states)
        # Q'(s')
        target_val = self.target_net.predict(next_states)

        for i in range(batch_size):
            if done[i]:
                target[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + self.gamma * (
                    target_val[i][a])

        self.online_net.fit(states, target, epochs=1,
                            verbose=0, callbacks=[self.tensorboard])
        self.target_net.set_weights(self.tau * np.array(self.online_net.get_weights()) + (
            1 - self.tau) * np.array(self.target_net.get_weights()))

        self.online_weights = self.online_net.get_weights()
        self.target_weights = self.target_net.get_weights()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Loads weights by name."""
        # TODO should load target_net
        self.online_net.load_weights(name)

    def save(self, name):
        """Saves weights by name."""
        # TODO should save target_net
        self.online_net.save_weights(name)
