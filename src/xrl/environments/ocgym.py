"""
Augmented gym that contain additional info based on ATARIARI wrapper module
"""
import gym
from termcolor import colored
try:
    from ocatari.core import OCAtari
<<<<<<< HEAD
except ImportError as imp_err:
=======
except ModuleNotFoundError as imp_err:
>>>>>>> focus file support first version
    print(colored("OC-Atari Not found, please install it:", "red"))
    print(colored("https://github.com/k4ntz/OC_Atari", "blue"))
    #print("test")
    exit()
    #raise imp_err


def make(env_name, notify=False):
    if notify:
        print(colored("Using AtariARI", "green"))
    return OCAtari(env_name, mode="revised")
