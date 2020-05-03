from config import MuZeroConfig, make_centipede_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.replay_buffer import ReplayBuffer
from training.training import train_network
import argparse
from signal import signal, SIGINT
from sys import exit
from time import time
# Disable WARNING logs to stdout
import tensorflow_core as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def muzero(config: MuZeroConfig, save_directory: str, load_directory: str, test: bool, visual: bool):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    """

    if load_directory is not None:
        # User specified directory to load network from
        network = config.old_network(load_directory)
    else:
        network = config.new_network()

    storage = SharedStorage(network, config.uniform_network(), config.new_optimizer(), save_directory, load_directory != None)
    replay_buffer = ReplayBuffer(config)

    if test:
        test_eps = 5
        eval_results = run_eval(config, storage, test_eps, visual=visual)
        print("Eval score: {} [{}]".format(eval_results[0], eval_results[1]))
        print(f"MuZero played {test_eps} "
              f"episodes.\n")
        return storage.latest_network()

    test_eps = 0
    best_score = 0
    for loop in range(config.nb_training_loop):
        start = time()
        print("Training loop", loop)

        score_train, scores = run_selfplay(config, storage, replay_buffer, config.nb_episodes, visual=visual)
        print("\tTrain score: " + str(score_train), '({} games: {})'.format(config.nb_episodes, scores))
        self_play_time = time()
        if score_train > best_score and loop > 0:
            storage.save_network_to_disk(storage.latest_network(), 'networks/instances/best-breakout')
            best_score = score_train

        train_network(config, storage, replay_buffer, config.nb_epochs)
        train_time = time()

        if loop % 10 == 0:
            eval_results = run_eval(config, storage, test_eps, visual=visual)
            print("Eval score: {} [{}]".format(eval_results[0], eval_results[1]))
            eval_time = time()

        print(f"\tMuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.")

        print("\tTime: {} total; {} selfplay; {} training; {} eval;".format(
            round(time() - start, 2),
            round(self_play_time - start, 2),
            round(train_time - self_play_time, 2),
            round(eval_time - train_time, 2)))

    return storage.latest_network()


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
    # parser.add_argument('-c', '--centipede',
    #                     action='store_true',
    #                     help="specify playing Centipede instead of CartPole")
    args = parser.parse_args()

    muzero(make_centipede_config(), args.save, args.load, args.test, args.visual)
