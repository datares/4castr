from collections import deque
import random
import numpy as np
from lib.agents.models.mlp import mlp


class DQNAgent(object):
    """ A simple Deep Q agent """

    def __init__(self, state_size, action_size):
        """
        Initializes a DQN agent.

        Args:
            state_size (int): Size of the state vector.
            action_size (int): Size of the action vector.
            mode (string): Model purpose
                (training, finetuning, validation, testing).

        Returns:
            None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.model = mlp(state_size, action_size)

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
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
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

        """ vectorized implementation; 30x speed up compared with for loop """
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a)
        target = rewards + self.gamma * \
            np.amax(self.model.predict(next_states), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model.predict(states)
        # make the agent to approximately map the current state to 
        # future discounted reward
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Loads weights by name."""
        self.model.load_weights(name)

    def save(self, name):
        """Saves weights by name."""
        self.model.save_weights(name)
