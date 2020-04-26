from config import MuZeroConfig, make_centipede_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval, multiprocess_play_game
from training.replay_buffer import ReplayBuffer
from training.training import train_network
import argparse
from signal import signal, SIGINT
from sys import exit
from time import time
import os
import tensorflow_core as tf
# Disable WARNING logs to stdout
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def muzero(config: MuZeroConfig, save_directory: str, load_directory: str, test: bool, visual: bool, new_config: bool):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    """
    config.load_directory = load_directory
    config.save_directory = save_directory
    replay_buffer = ReplayBuffer(config)
    # Remove old checkpoint network
    base_dir = os.path.dirname(os.path.realpath(__file__))
    d = base_dir + '/checkpoint'
    to_remove = [os.path.join(d, f) for f in os.listdir(d)]
    for f in to_remove:
        if f.split('/')[-1] != '.gitignore':
            os.remove(f)

    if new_config:
        network = config.new_network()
        SharedStorage.save_network_to_disk(network, config, None, 'blank_network')
        exit(0)

    if test:
        if load_directory is not None:
            # User specified directory to load network from
            network = config.old_network(load_directory)
        else:
            network = config.new_network()
        storage = SharedStorage(network, config.uniform_network(), config.new_optimizer(), save_directory, config, load_directory != None)
        # Single process for simple testing, can refactor later
        print("Eval score:", run_eval(config, storage, 5, visual=visual))
        print(f"MuZero played {5} "
              f"episodes.\n")
        return storage.latest_network()

    for loop in range(config.nb_training_loop):
        initial = True if loop == 0 else False
        start = time()
        o_start = time()
        print("Training loop", loop)
        episodes = config.nb_episodes

        score_train = multiprocess_play_game(config, initial=initial, episodes=episodes, train=True, replay_buffer=replay_buffer)
        print("Self play took " + str(time() - start) + " seconds")
        print("Train score: " + str(score_train) + " after " + str(time() - start) + " seconds")

        start = time()
        print("Training network...")
        train_network(config, replay_buffer, config.nb_epochs)
        print("Network weights updated after " + str(time() - start) + " seconds")
        """

        start = time()
        eval_eps = 5
        print("Evaluating model...")
        score_eval = multiprocess_play_game(config, initial=False, episodes=eval_eps, train=False, replay_buffer=replay_buffer)
        print("Eval score: " + str(score_eval) + " after " + str(time() - start) + " seconds and " + str(eval_eps) + " episodes")
        print(f"MuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")
        print("Total loop time: " + str(time() - o_start) + " seconds")
        """


def handler(signal_received, frame):
    print('\nUser terminated program. Goodbye!')
    exit(0)


if __name__ == '__main__':
    signal(SIGINT, handler)

    parser = argparse.ArgumentParser(description='Run MuZero')
    parser.add_argument('-l', '--load',
                        type=str,
                        help='directory to load network from')
    parser.add_argument('-s', '--save',
                        type=str,
                        help='directory to save network to')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='only run the specified network, do not train')
    parser.add_argument('-v', '--visual',
                        action='store_true',
                        help="display the network's evaluation for user")
    parser.add_argument('-n', '--new_config',
                        action='store_true',
                        help="reload a new blank network if config has been changed")
    args = parser.parse_args()

    muzero(make_centipede_config(), args.save, args.load, args.test, args.visual, args.new_config)
