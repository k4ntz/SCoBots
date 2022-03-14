# file for extracting raw features with atariari

import numpy as np
import math

# function to return the next saw in ecoinrun
def get_next_saw(env_info):
    player_x = env_info["agent_pos"][0]
    # check all saws
    max_saw_count = 5
    next_saws = []
    for i in range(1, max_saw_count + 1):
        saw = env_info["saw" + str(i) + "_pos"]
        # return current saw because it is the closest one still coming
        if saw[0] > player_x:
            return saw
    # when no saw available
    return env_info["saw1_pos"]


# helper function for bowling
def add_pin_coords(xs, ys, x, y):
    xs.append(x)
    ys.append(y)
    return xs, ys

# helper function to get average position of bowling pin
def get_avg_bowling_pin(labels):
    xs = []
    ys = []
    # hard coded since its quite weird whats written in atariari for it
    if labels["pin_existence_0"] < 200:
        xs, ys = add_pin_coords(xs, ys, 103, 15)
    if labels["pin_existence_1"] < 200:
        xs, ys = add_pin_coords(xs, ys, 107, 18)
    if labels["pin_existence_2"] < 200:
        xs, ys = add_pin_coords(xs, ys, 107, 12)
    if labels["pin_existence_3"] < 200:
        xs, ys = add_pin_coords(xs, ys, 111, 21)
    if labels["pin_existence_4"] < 200:
        xs, ys = add_pin_coords(xs, ys, 111, 15)
    if labels["pin_existence_5"] < 200:
        xs, ys = add_pin_coords(xs, ys, 111, 9)
    if labels["pin_existence_6"] < 200:
        xs, ys = add_pin_coords(xs, ys, 115, 25)
    if labels["pin_existence_7"] < 200:
        xs, ys = add_pin_coords(xs, ys, 115, 18)
    if labels["pin_existence_8"] < 200:
        xs, ys = add_pin_coords(xs, ys, 115, 12)
    if labels["pin_existence_9"] < 200:
        xs, ys = add_pin_coords(xs, ys, 115, 5)
    # calc avg pos
    x = 0
    y = 0
    # calc mean when min. one pin is standing
    if len(xs) > 0:
        x = sum(xs) / len(xs)
        y = sum(ys) / len(ys)
    return [np.int16(x), np.int16(y)]


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
        saw = get_next_saw(env_info)
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, coin, saw, None, None, None]
        else:
            raw_features = np.roll(raw_features, 3)
            raw_features[0] = player
            raw_features[1] = coin
            raw_features[2] = saw
        return raw_features
    ###########################################
    # bowling game
    elif gametype == 4:
        player = [labels["player_x"].astype(np.int16),
                labels["player_y"].astype(np.int16)]
        pin = get_avg_bowling_pin(labels)
        ball = [labels["ball_x"].astype(np.int16),
                labels["ball_y"].astype(np.int16)]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, pin, ball, None, None, None]
        else:
            raw_features = np.roll(raw_features, 3)
            raw_features[0] = player
            raw_features[1] = pin
            raw_features[2] = ball
        return raw_features