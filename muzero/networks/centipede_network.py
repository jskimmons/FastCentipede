import math

import numpy as np
from .convolutional_networks import *
from tensorflow_core.python.keras import regularizers

from game.game import Action
from networks.network import BaseNetwork


class CentipedeNetwork(BaseNetwork):

    def __init__(self,
                 action_size: int,
                 representation_size: (int, int),
                 max_value: int,
                 weight_decay: float = 1e-4,
                 directory: str = None):
        self.representation_size = representation_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

        if directory is not None:
            print("Loading network from " + directory)
            representation_network = self.load_model(directory + "/representation")
            value_network = self.load_model(directory + "/value")
            policy_network = self.load_model(directory + "/policy")
            dynamic_network = self.load_model(directory + "/dynamic")
            reward_network = self.load_model(directory + "/reward")
        else:
            print("Creating new network")
            regularizer = regularizers.l2(weight_decay)

            representation_network = build_representation_network(representation_size)

            # Ignore batch size when setting network inputs
            hidden_rep_shape = representation_network.output_shape[1:]
            value_network = build_value_network(hidden_rep_shape, self.value_support_size)
            policy_network = build_policy_network(hidden_rep_shape, self.action_size, regularizer)

            # Shape when actions are stacked on top of hidden rep
            stacked_hidden_rep_shape = (hidden_rep_shape[0], hidden_rep_shape[1], hidden_rep_shape[2] + 1)
            dynamic_network = build_dynamic_network(stacked_hidden_rep_shape)
            reward_network = build_reward_network(stacked_hidden_rep_shape)

            """
            representation_network.summary()
            value_network.summary()
            policy_network.summary()
            dynamic_network.summary()
            reward_network.summary()
            """

        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.asscalar(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        one_hot_action = np.eye(self.action_size)[action.index]
        one_hot_action = np.reshape(one_hot_action, (1, 3, 1))
        concat_action = np.zeros((hidden_state.shape[0], hidden_state.shape[1], 1))
        concat_action[:one_hot_action.shape[0], :one_hot_action.shape[1]] = one_hot_action

        conditioned_hidden = np.concatenate((hidden_state, concat_action), axis=2)
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)
