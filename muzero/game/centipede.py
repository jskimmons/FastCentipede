from typing import List

import gym

from game.game import Action, AbstractGame
from game.gym_wrappers import DownSampleVisualObservationWrapper, LimitToShootActions

class Centipede(AbstractGame):
    """The Gym Centipede environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make('Breakout-v0')
        #self.env = LimitToShootActions(self.env)
        self.env = DownSampleVisualObservationWrapper(self.env, factor=5)
        self.actions = list(map(lambda i: Action(i), list(range(self.env.action_space.n))))
        self.observations = [self.env.reset()]
        self.done = False
        self.curr_lives = 5
        self.env.step(1)

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""
        if action.index < 2:
            action.index = 2
        elif action.index >= 2:
            action.index = 3
        observation, reward, done, info = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        if info['ale.lives'] < self.curr_lives and not done:
            # Make new ball fire
            self.curr_lives = info['ale.lives']
            observation, reward, done, info = self.env.step(1)
            self.done = done

        return reward

    def terminal(self) -> bool:
        """Is the game is finished?"""
        if self.curr_lives < 5:
            return True
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]
