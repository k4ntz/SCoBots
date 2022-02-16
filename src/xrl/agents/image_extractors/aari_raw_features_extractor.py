# file for extracting raw features with atariari

import numpy as np
import math

# function to get raw features and order them by
def get_raw_features(env_info, last_raw_features=None, gametype=0):
    # extract raw features
    if gametype != 3:
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
    ###########################################
    # coinrun
    elif gametype == 3:
        player = env_info["agent_pos"]
        coin = env_info["coin_pos"]
        saw1 = env_info["saw1_pos"]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, coin, saw1, None, None, None]
        else:
            raw_features = np.roll(raw_features, 3)
            raw_features[0] = player
            raw_features[1] = coin
            raw_features[2] = saw1
        return raw_features