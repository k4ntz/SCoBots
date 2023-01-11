"""
Augmented gym that contain additional info based on ATARIARI wrapper module
"""
import gymnasium as gym
from termcolor import colored
try:
    from ocatari.core import OCAtari
except ImportError as imp_err:
    print(colored("OC-Atari Not found, please install it:", "red"))
    print(colored("https://github.com/k4ntz/OC_Atari", "blue"))
    #print("test")
    exit()
    #raise imp_err


def make(env_name, notify=False):
    if notify:
        print(colored("Using AtariARI", "green"))
    return OCAtari(env_name, mode="revised")
