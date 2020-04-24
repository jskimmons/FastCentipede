from config import MuZeroConfig
from tensorflow_core.keras import optimizers
from networks.network import BaseNetwork, UniformNetwork, AbstractNetwork, InitialModel, RecurrentModel
from copy import copy


class SharedStorage(object):
    """Save the different versions of the network."""

    def __init__(self, network: BaseNetwork, uniform_network: UniformNetwork, optimizer: optimizers,
                 save_directory: str, config: MuZeroConfig, pretrained: bool = False):
        self._networks = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.optimizer = optimizer
        self.directory = save_directory
        self.config = config
        self.checkpoint_directory = "checkpoint"

        if pretrained:
            self._networks[0] = network

    def latest_network(self) -> AbstractNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.uniform_network

    def save_network(self, step: int, network: BaseNetwork):
        self._networks[step] = network

    @staticmethod
    def save_network_to_disk(network: BaseNetwork, config):
        if config.save_directory:
            network.save_network(config.save_directory)
        network.save_network('checkpoint')

    def latest_network_for_process(self) -> AbstractNetwork:
        if self._networks:
            # Need to load network from disk independently for each individual process b/c tensorflow does not support
            # multiprocessing out of the box
            args = copy(self.config.network_args)
            args['directory'] = self.checkpoint_directory
            return self.config.network(**args)
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.uniform_network
