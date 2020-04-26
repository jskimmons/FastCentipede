import collections
from typing import Optional, Dict

import tensorflow_core as tf
from game.centipede import Centipede
from game.game import AbstractGame
from networks.centipede_network import CentipedeNetwork
from networks.network import BaseNetwork, UniformNetwork

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig(object):

    def __init__(self,
                 game,
                 nb_training_loop: int,
                 nb_episodes: int,
                 nb_epochs: int,
                 network_args: Dict,
                 network,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 visit_softmax_temperature_fn,
                 lr: float,
                 known_bounds: Optional[KnownBounds] = None):
        ### Environment
        self.game = game

        ### Self-Play
        self.action_space_size = action_space_size
        # self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.nb_training_loop = nb_training_loop
        self.nb_episodes = nb_episodes  # Nb of episodes per training loop
        self.nb_epochs = nb_epochs  # Nb of epochs per training loop

        # self.training_steps = int(1000e3)
        # self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 0
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.network_args = network_args
        self.network = network
        self.lr = lr
        # Exponential learning rate schedule
        # self.lr_init = lr_init
        # self.lr_decay_rate = 0.1
        # self.lr_decay_steps = lr_decay_steps

    def new_game(self) -> AbstractGame:
        return self.game(self.discount)

    def new_network(self) -> BaseNetwork:
        return self.network(**self.network_args)

    def old_network(self, directory) -> BaseNetwork:
        self.network_args['directory'] = directory
        return self.network(**self.network_args)

    def uniform_network(self) -> UniformNetwork:
        return UniformNetwork(self.action_space_size)

    def new_optimizer(self) -> tf.keras.optimizers:
        return tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum)

def make_centipede_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        game=Centipede,
        nb_training_loop=3,
        nb_episodes=5,
        nb_epochs=10,
        network_args={'action_size': 18,
                      'state_size': 4,
                      'representation_size': (50, 32, 1),
                      'max_value': 5000},
        network=CentipedeNetwork,
        action_space_size=18,
        max_moves=20000,
        discount=0.3,
        dirichlet_alpha=0.25,
        num_simulations=11,  # Odd number perform better in eval mode
        batch_size=512,
        td_steps=10,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        lr=0.05)
"""
Legacy configs from the DeepMind's pseudocode.
def make_atari_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature)
"""
