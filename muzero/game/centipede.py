from typing import List

import gym

from game.game import Action, AbstractGame
from game.gym_wrappers import DownSampleVisualObservationWrapper
from itertools import chain

class Centipede(AbstractGame):
    """The Gym Centipede environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make('Centipede-v0')
        self.env = DownSampleVisualObservationWrapper(self.env, factor=5)
        self.actions = list(map(lambda i: Action(i), list(range(9))))
        self.observations = [self.env.reset()]
        self.done = False
        self.current_lives = 3

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""
        # Remap to only use shooting actions
        step_index = 1 if action.index == 0 else action.index + 9
        observation, reward, done, info = self.env.step(step_index)
        num_lives = info['ale.lives']
        self.observations += [observation]
        self.done = done
        if num_lives < self.current_lives:
            self.current_lives = num_lives
            # Might be messing up some gradient stuff when reward gets negative. Not sure yet
            #reward = -1000
        return reward

    def terminal(self) -> bool:
        """Is the game is finished?"""
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]
