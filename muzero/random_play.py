import gym
import random
import numpy as np

env = gym.make('Centipede-v0')
scores = list()

for i in range(30):
    env.reset()
    score = 0
    done = False
    while not done:
        action = int(random.randrange(0, 18, 1))
        observation, reward, done, _ = env.step(action)
        score += reward
        # env.render()
    print("Game " + str(i+1) + ": " + str(score))
    scores.append(score)

avg = np.average(np.asarray(scores))
print(avg)

