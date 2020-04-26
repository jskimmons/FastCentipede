from config import MuZeroConfig
from tensorflow_core.keras import optimizers
import tensorflow_core.keras.backend as K
#from tensorflow_core.keras.saving.hdf5_format import save_optimizer_weights_to_hdf5_group
from networks.network import BaseNetwork, UniformNetwork, AbstractNetwork, InitialModel, RecurrentModel
from copy import copy
#import h5py
import pickle


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
    def save_optimizer_to_disk(optimizer, dir):
        """
        f = h5py.File(dir+'/optimizer.h', mode='w')
        symbolic_weights = getattr(optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = f.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
        if optimizer:
            save_optimizer_weights_to_hdf5_group(f, model.optimizer)
        """
        if optimizer is None:
            return
        symbolic_weights = getattr(optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(dir + '/optimizer.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

    @staticmethod
    def save_network_to_disk(network: BaseNetwork, config, optimizer, file=None):
        if file:
            network.save_network('blank_network')
            SharedStorage.save_optimizer_to_disk(optimizer, 'blank_network')
        if config.save_directory:
            network.save_network(config.save_directory)
            SharedStorage.save_optimizer_to_disk(optimizer, config.save_directory)
        network.save_network('checkpoint')
        SharedStorage.save_optimizer_to_disk(optimizer, 'checkpoint')

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
