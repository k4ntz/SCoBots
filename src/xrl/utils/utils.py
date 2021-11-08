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
from xrl.xrl_config import cfg

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
##### PROCESSING FEATURES #####
###############################


# function to get raw features and order them by
def get_raw_features(env_info, last_raw_features=None, gametype=0):
    # extract raw features
    labels = env_info["labels"]
    # if ball game
    if gametype == 0:
        player = [labels["player_x"].astype(np.int16),
                labels["player_y"].astype(np.int16)]
        enemy = [labels["enemy_x"].astype(np.int16),
                labels["enemy_y"].astype(np.int16)]
        ball = [labels["ball_x"].astype(np.int16),
                labels["ball_y"].astype(np.int16)]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, enemy, ball, None, None, None]
        else:
            raw_features = np.roll(raw_features, 3)
            raw_features[0] = player
            raw_features[1] = enemy
            raw_features[2] = ball
        return raw_features
    ###########################################
    # demon attack game
    elif gametype == 1:
        player = [labels["player_x"].astype(np.int16),
                np.int16(3)]        # constant 3
        enemy1 = [labels["enemy_x1"].astype(np.int16),
                labels["enemy_y1"].astype(np.int16)]
        enemy2 = [labels["enemy_x2"].astype(np.int16),
                labels["enemy_y2"].astype(np.int16)]
        enemy3 = [labels["enemy_x3"].astype(np.int16),
                labels["enemy_y3"].astype(np.int16)]
        #missile = [labels["player_x"].astype(np.int16),
        #        labels["missile_y"].astype(np.int16)]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, enemy1, enemy2, enemy3, None, None, None, None]
        else:
            raw_features = np.roll(raw_features, 4)
            raw_features[0] = player
            raw_features[1] = enemy1
            raw_features[2] = enemy2
            raw_features[3] = enemy3
        return raw_features
    ###########################################
    # boxing game
    elif gametype == 2:
        player = [labels["player_x"].astype(np.int16),
                labels["player_y"].astype(np.int16)]
        enemy = [labels["enemy_x"].astype(np.int16),
                labels["enemy_y"].astype(np.int16)]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, enemy, None, None]
        else:
            raw_features = np.roll(raw_features, 2)
            raw_features[0] = player
            raw_features[1] = enemy
        return raw_features


# function to get integrated gradients
def get_integrated_gradients(ig, input, target_class):
    # get attributions and print
    attributions, approximation_error = ig.attribute(input,
        target=target_class, return_convergence_delta=True)
    #print(attributions)
    attr = attributions[0].cpu().detach().numpy()
    #print(attr_df)
    return attr


# helper function to calc linear equation
def get_lineq_param(obj1, obj2):
    x = obj1
    y = obj2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


# helper function to convert env info into custom list
# raw_features contains player x, y, ball x, y, oldplayer x, y, oldball x, y,
# features are processed stuff for policy
def preprocess_raw_features(raw_features):
    n_raw_features = int(len(raw_features)/2)
    features = []
    for i in range(0, n_raw_features):
        obj1, obj1_past = raw_features[i], raw_features[i + n_raw_features]
        # when object has moved and has history
        if obj1_past is not None and not (obj1[0] == obj1_past[0] and obj1[1] == obj1_past[1]):
            # append velocity of itself
            features.append(math.sqrt((obj1_past[0] - obj1[0])**2 + (obj1_past[1] - obj1[1])**2))
        else:
            features.append(0)
        for j in range(0, n_raw_features):
            # apped all manhattan distances to all other objects
            # which are not already calculated
            if j > i:
                obj2 = raw_features[j]
                # append coord distances
                features.append(obj2[0] - obj1[0]) # append x dist
                features.append(obj2[1] - obj1[1]) # append y dist
        for j in range(0, n_raw_features):
            # calculate movement paths of all other objects
            # and calculate distance to its x and y intersection
            if i != j:
                obj2, obj2_past = raw_features[j], raw_features[j + n_raw_features]
                # if other object has moved
                if obj2_past is not None and not (obj2[0] == obj2_past[0] and obj2[1] == obj2_past[1]):
                    # append trajectory cutting points
                    m, c = get_lineq_param(obj2, obj2_past)
                    # now calc target pos
                    # y = mx + c substracted from its y pos
                    features.append(np.int16(m * obj1[0] + c) - obj1[1])
                    # x = (y - c)/m substracted from its x pos
                    features.append(np.int16((obj1[1] - c) / m)  - obj1[0])
                else:
                    features.append(0)
                    features.append(0)
    return raw_features, features


# helper function to get features
def do_step(env, action=1, last_raw_features=None):
    obs, reward, done, info = env.step(action)
    # check if ball game
    # 0: ball game
    # 1: demonattack
    # 2: boxing
    name = env.unwrapped.spec.id
    gametype = 0
    if "Demon" in name:
        gametype = 1
    elif "Boxing" in name:
        gametype = 2
    # calculate meaningful features from given raw features and last raw features
    raw_features = get_raw_features(info, last_raw_features, gametype=gametype)
    raw_features, features = preprocess_raw_features(raw_features)
    return raw_features, features, reward, done


# do 5 episodes and get features to prune from corr matrix
def init_corr_prune(env, it = 5, tr = 0.75):
    features_list = []
    # run it episodes to collect features data
    for i in tqdm(range(it)):
        n_actions = env.action_space.n
        _ = env.reset()
        _, _, done, _ = env.step(1)
        raw_features, features, _, _ = do_step(env)
        for t in range(50000):  # max 50k steps per episode
            action = np.random.randint(0, n_actions)
            raw_features, features, reward, done = do_step(env, action, raw_features)
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
