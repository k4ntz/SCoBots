from operator import contains
import os
import argparse
import time
import numpy as np

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.decomposition import PCA
from scipy.stats import entropy

from argparse import ArgumentParser
from utils.xrl_config import cfg
from functools import wraps
from termcolor import colored

# player enemy ball
features_names = [
    "0: player speed",
    "1: x enemy - player",
    "2: y enemy - player",
    "3: x ball - player",
    "4: y ball - player",
    "5: y enemy-target - player",
    "6: x enemy-target - player",
    "7: y ball-target - player",
    "8: x ball-target - player",
    "9: enemy speed",
    "10: x ball - enemy",
    "11: y ball - enemy",
    "12: y player-target - enemy",
    "13: x player-target - enemy",
    "14: y ball-target - enemy",
    "15: x ball-target - enemy",
    "16: ball speed",
    "17: y player-target - ball",
    "18: x player-target - ball",
    "19: y enemy-target - ball",
    "20: x enemy-target - ball",
]

######################
######## INIT ########
######################

# function to get config
def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )

    parser.add_argument(
        '--space-config-file',
        type=str,
        default='configs/atari_ball_joint_v1.yaml',
        metavar='FILE',
        help='Path to SPACE config file'
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')

    # Seed
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return cfg


###############################
######## UTILS STUFF ##########
###############################

# calc single entropy of one feature
def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)


# calc entropy of all features given
def calc_entropies(features):
    features = np.array(features)
    entropies = []
    for col in features.T:
        entropies.append(entropy1(col))
    print(entropies)
    return entropies


# TODO: Replace with external file or sth similar
# get gametype
# check if ball game
# 0: ball game
# 1: demonattack
# 2: boxing    
# 3: coinrun
# 4: bowling
def get_gametype(env):
    name = env._env.unwrapped.spec.id
    gametype = 0
    if "Demon" in name:
        gametype = 1
    elif "Boxing" in name:
        gametype = 2
    elif "coin" in name:
        gametype = 3
    elif "Bowling" in name:
        gametype = 4
    elif "Skiing" in name:
        gametype = 5
    return gametype


# timing wrapper
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"function: '{func.__name__}' | duration: {total_time:.4f}s")
        return result
    return timeit_wrapper


def color_me(new, old):
    if new > old:
            col = "light_green"
    else:
            col = "light_red"
    return colored("{:.2f}".format(new), col)
