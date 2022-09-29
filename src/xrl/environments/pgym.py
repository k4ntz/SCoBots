"""
Augmented gym that contain additional info based on procgen wrapper module
"""
import numpy as np
import sys
import gym
from termcolor import colored
try:
    from utils_procgen import InteractiveEnv
    from procgen import ProcgenGym3Env
except ImportError as imp_err:
    print(colored("ProcgenGym Not found, please install it:", "red"))
    print(colored("https://github.com/k4ntz/procgen", "blue"))
    raise imp_err


def make(env_name, notify=False):
    if notify:
        print(colored("Using ProcgenGymEnv", "green"))
    return gym.make("procgen:procgen-" + env_name + "-v0")