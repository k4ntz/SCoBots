"""
Augmented gym that contain additional info based on ATARIARI wrapper module
"""
import gymnasium as gym
from termcolor import colored
from scobi.environments.hackatari.core import HackAtari
try:
    from ocatari.core import OCAtari
except ImportError as imp_err:
    print(colored("OC-Atari Not found, please install it:", "red"))
    print(colored("https://github.com/k4ntz/OC_Atari", "blue"))
    #print("test")
    exit()
    #raise imp_err


def make(env_name, useHacks=False, mods=None, notify=False, *args, **kwargs):
    if mods is None:
        mods = []
    if notify:
        print(colored("Using AtariARI", "green"))
    print(env_name)
    if useHacks:
        print('Using HackAtari as environment')
        return HackAtari(difficulty=0, env_name=env_name,
                         game_mode=0,modifs=mods,
                         switch_frame=0, render_mode="rgb_array",
                         obs_mode="obj",
                         mode="ram",
                         render_oc_overlay=True,
                         frameskip=4, *args, **kwargs)
    else: return OCAtari(env_name, "ram", *args, **kwargs) 
