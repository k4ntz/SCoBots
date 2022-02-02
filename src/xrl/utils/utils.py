from operator import contains
import os
import math
import torch
import cv2
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import seaborn as sns

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import entropy

from argparse import ArgumentParser
from xrl.utils.xrl_config import cfg

import xrl.agents.env_steps as env_steps

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

# function to get integrated gradients
def get_integrated_gradients(ig, input, target_class):
    # get attributions and print
    attributions, approximation_error = ig.attribute(input,
        target=target_class, return_convergence_delta=True)
    #print(attributions)
    attr = attributions[0].cpu().detach().numpy()
    #print(attr_df)
    return attr


# do 5 episodes and get features to prune from corr matrix
def init_corr_prune(env, it = 5, tr = 0.75):
    features_list = []
    # run it episodes to collect features data
    for i in tqdm(range(it)):
        n_actions = env.action_space.n
        _ = env.reset()
        _, _, done, _ = env.step(1)
        raw_features, features, _, _ = env_steps.do_step(env)
        for t in range(50000):  # max 50k steps per episode
            action = np.random.randint(0, n_actions)
            raw_features, features, reward, done = env_steps.do_step(env, action, raw_features)
            features_list.append(features)
            if done:
                break
    # now make corr matrix
    df = pd.DataFrame(data=features_list)
    corr = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # Find features with correlation greater than tr
    to_drop = [column for column in upper.columns if any(upper[column] > tr)]
    return to_drop


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


# get gametype
# check if ball game
# 0: ball game
# 1: demonattack
# 2: boxing    
def get_gametype(env):
    name = env.unwrapped.spec.id
    gametype = 0
    if "Demon" in name:
        gametype = 1
    elif "Boxing" in name:
        gametype = 2
    return gametype
