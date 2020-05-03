import gym
import random
import numpy as np
from game.centipede import Centipede
from game.game import Action
game = Centipede(discount=0.997)
scores = list()

for i in range(3):
    game = Centipede(discount=0.997)
    score = 0
    done = False
    while not game.terminal():
    #for i in range(200):
        action = Action(int(np.random.choice([0,1])))
        print(action.index)
        reward = game.step(action)
        score += reward
        game.env.render()
    print("Game " + str(i+1) + ": " + str(score))
    scores.append(score)
    game.env.close()

avg = np.average(np.asarray(scores))
print(avg)

