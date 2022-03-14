from cmath import inf
import gym
import numpy as np
import os
import random

import matplotlib.pyplot as plt

from atariari.benchmark.wrapper import AtariARIWrapper

# YarsRevenge
# 
env_name = "BowlingDeterministic-v4"

def print_labels(env_info):
    # extract raw features
    labels = env_info["labels"]
    ypos = labels["ball_y"]
    if ypos > 5 and ypos < 115:
        None
        print(labels)

env = AtariARIWrapper(gym.make(env_name))
name = env.unwrapped.spec.id
print(name)

n_actions = env.action_space.n
_ = env.reset()
obs, _, done, info = env.step(0)

r = 0
for t in range(50000):
    plt.imshow(obs, interpolation='none')
    plt.plot()
    plt.pause(0.0001)  # pause a bit so that plots are updated
    action = random.randint(0, n_actions - 1)
    obs, reward, done, info = env.step(action)
    r += reward
    print("Reward:", reward)
    print_labels(info)
    print("------")
    if(done):
        break
print(r)