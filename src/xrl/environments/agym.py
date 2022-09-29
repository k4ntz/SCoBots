"""
Augmented gym that contain additional info based on ATARIARI wrapper module
"""
import gym
from termcolor import colored
try:
    from atariari.benchmark.wrapper import AtariARIWrapper
except ImportError as imp_err:
    print(colored("AtariARI Not found, please install it:", "red"))
    print(colored("https://github.com/mila-iqia/atari-representation-learning:", "blue"))
    raise imp_err


def make(env_name, notify=False):
    if notify:
        print(colored("Using AtariARI", "green"))
    return AtariARIWrapper(gym.make(env_name))
