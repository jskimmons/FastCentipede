"""Training module: this is where MuZero neurons are trained."""
import numpy as np
from networks.shared_storage import SharedStorage
import multiprocessing
from config import MuZeroConfig
from networks.network import BaseNetwork
from tensorflow_core.keras import optimizers
from training.replay_buffer import ReplayBuffer
from tensorflow_core.python.keras.losses import MSE
from tensorflow_core import stop_gradient, boolean_mask, one_hot, reshape, constant, pad, concat, convert_to_tensor, squeeze
from tensorflow_core.math import reduce_mean
from tensorflow_core.nn import softmax_cross_entropy_with_logits


def train_network_helper(config: MuZeroConfig, replay_buffer: ReplayBuffer, epochs: int):
    try:
        network = config.old_network('checkpoint')
    except FileNotFoundError:
        network = config.new_network()

    optimizer = config.new_optimizer()

    for _ in range(epochs):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch)
    SharedStorage.save_network_to_disk(network, config)


def train_network(config: MuZeroConfig, replay_buffer: ReplayBuffer, epochs:int):
    """
    Tensorflow freaks out when multiprocessing happens. Need to have all instantiations of networks etc. In child
    processes or else network operations will hang forever
    """
    train_process = multiprocessing.Process(target=train_network_helper, args=(config, replay_buffer, epochs))
    train_process.start()
    train_process.join()


def update_weights(optimizer: optimizers, network: BaseNetwork, batch):

    def scale_gradient(tensor, scale: float):
        """Trick function to scale the gradient in tensorflow"""
        return (1. - scale) * stop_gradient(tensor) + scale * tensor

    def loss():
        loss = 0
        image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch = batch

        # Initial step, from the real observation: representation + prediction networks
        representation_batch, value_batch, policy_batch = network.initial_model(np.array(image_batch))

        # Only update the element with a policy target
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch)
        mask_policy = list(map(lambda l: bool(l), target_policy_batch))
        target_policy_batch = list(filter(lambda l: bool(l), target_policy_batch))
        policy_batch = boolean_mask(policy_batch, mask_policy)

        # Compute the loss of the first pass
        loss += reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size))
        loss += reduce_mean(
            softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch))

        # Recurrent steps, from action and previous hidden state.
        for actions_batch, targets_batch, mask, dynamic_mask in zip(actions_time_batch, targets_time_batch,
                                                                    mask_time_batch, dynamic_mask_time_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            # Only execute BPTT for elements with an action
            representation_batch = boolean_mask(representation_batch, dynamic_mask)
            target_value_batch = boolean_mask(target_value_batch, mask)
            target_reward_batch = boolean_mask(target_reward_batch, mask)

            # Creating conditioned_representation: concatenate representations with actions batch
            actions_batch = one_hot(actions_batch, network.action_size)

            # TODO: make this reshape dynamic
            actions_batch = reshape(actions_batch, (actions_batch.shape[0], 6, 3, 1))

            paddings = constant([[0, 0],
                                    [0, max(0, representation_batch.shape[1] - actions_batch.shape[1])],
                                    [0, max(0, representation_batch.shape[2] - actions_batch.shape[2])],
                                    [0, 0]])
            actions_batch = pad(actions_batch, paddings, "CONSTANT")

            # Recurrent step from conditioned representation: recurrent + prediction networks
            conditioned_representation_batch = concat((representation_batch, actions_batch), axis=3)
            representation_batch, reward_batch, value_batch, policy_batch = network.recurrent_model(
                conditioned_representation_batch)

            # Only execute BPTT for elements with a policy target
            target_policy_batch = [policy for policy, b in zip(target_policy_batch, mask) if b]
            mask_policy = list(map(lambda l: bool(l), target_policy_batch))
            target_policy_batch = convert_to_tensor([policy for policy in target_policy_batch if policy])
            policy_batch = boolean_mask(policy_batch, mask_policy)

            # Compute the partial loss
            l = (reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size)) +
                 MSE(target_reward_batch, squeeze(reward_batch)) +
                 reduce_mean(
                     softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch)))

            # Scale the gradient of the loss by the average number of actions unrolled
            gradient_scale = 1. / len(actions_time_batch)
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient of the representation
            representation_batch = scale_gradient(representation_batch, 0.5)

        return loss

    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    network.training_steps += 1


def loss_value(target_value_batch, value_batch, value_support_size: int):
    batch_size = len(target_value_batch)
    targets = np.zeros((batch_size, value_support_size))
    sqrt_value = np.sqrt(target_value_batch)
    floor_value = np.floor(sqrt_value).astype(int)
    floor_value = np.clip(floor_value, a_min=0, a_max=value_support_size-2)
    rest = sqrt_value - floor_value
    targets[range(batch_size), floor_value.astype(int)] = 1 - rest
    targets[range(batch_size), floor_value.astype(int) + 1] = rest

    return softmax_cross_entropy_with_logits(logits=value_batch, labels=targets)
