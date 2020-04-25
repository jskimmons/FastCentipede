"""Self-Play module: where the games are played."""

from config import MuZeroConfig
from game.game import AbstractGame
from networks.shared_storage import SharedStorage
from self_play.mcts import run_mcts, select_action, expand_node, add_exploration_noise
from self_play.utils import Node
from training.replay_buffer import ReplayBuffer
from time import time
import multiprocessing
from multiprocessing import Queue, Pool, Semaphore, Process
import os


def multiprocess_play_game_helper(config: MuZeroConfig, initial: bool, train: bool, result_queue: Queue = None, sema=None):
    sema.acquire()
    # Prevent child processes from overallocating GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    pretrained = True
    if initial:
        if config.load_directory is not None:
            # User specified directory to load network from
            network = config.old_network(config.load_directory)
        else:
            network = config.old_network('blank_network')
            pretrained = False
    else:
        network = config.old_network('checkpoint')

    storage = SharedStorage(network=network, uniform_network=config.uniform_network(), optimizer=config.new_optimizer(),
                            save_directory=config.save_directory, config=config, pretrained=pretrained)

    play_game(config=config, storage=storage, train=train, visual=False, queue=result_queue)
    sema.release()


def multiprocess_play_game(config: MuZeroConfig, initial: bool, episodes: int, train: bool, replay_buffer: ReplayBuffer):
    """
    Plays game in a multiprocessed manner and returns average score over all episodes. Use for self-play and eval
    """

    result_queue = Queue()

    concurrency = multiprocessing.cpu_count()
    total_task_num = episodes
    sema = Semaphore(concurrency)
    all_processes = []
    for i in range(total_task_num):
        p = Process(target=multiprocess_play_game_helper, args=(config, initial, train, result_queue, sema))
        all_processes.append(p)
        p.start()

    games = [result_queue.get() for _ in range(episodes)]

    returns = []
    for game in games:
        replay_buffer.save_game(game)
        returns.append(sum(game.rewards))

    return sum(returns) / episodes if episodes else 0


def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, train_episodes: int):
    """Take the latest network, produces multiple games and save them in the shared replay buffer"""
    returns = []
    for _ in range(train_episodes):
        game = play_game(config, storage)
        replay_buffer.save_game(game)
        returns.append(sum(game.rewards))
    return sum(returns) / train_episodes


def run_eval(config: MuZeroConfig, storage: SharedStorage, eval_episodes: int, visual: bool = False):
    """Evaluate MuZero without noise added to the prior of the root and without softmax action selection"""
    returns = []
    for _ in range(eval_episodes):
        game = play_game(config, storage, train=False, visual=visual)
        returns.append(sum(game.rewards))
    return sum(returns) / eval_episodes if eval_episodes else 0


def play_game(config: MuZeroConfig, storage: SharedStorage, train: bool = True, visual: bool = False, queue: Queue = None) -> AbstractGame:
    """
    Each game is produced by starting at the initial board position, then
    repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    of the game is reached.
    """
    if queue:
        network = storage.latest_network_for_process()
    else:
        network = storage.current_network

    start = time()
    game = config.new_game()
    mode_action_select = 'softmax' if train else 'max'
    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(), network.initial_inference(current_observation))
        if train:
            add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network, mode=mode_action_select)
        game.apply(action)
        game.store_search_statistics(root)
        if visual:
            game.env.render()
    if visual:
        if game.terminal():
            print('Model lost game')
        else:
            print('Exceeded max moves')
        game.env.close()

    if queue:
        queue.put(game)
    print("Finished game episode after " + str(time() - start) + " seconds. Exceeded max moves? " + str(not game.terminal()))
    print("Score: ", sum(game.rewards))
    return game
