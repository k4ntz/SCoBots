# file for raw feature processing to meaningful features
# with atari ari gym environments

import numpy as np
import math

from scipy.spatial import KDTree
from webcolors import CSS2_HEX_TO_NAMES, hex_to_rgb


# helper function to calc linear equation
def get_lineq_param(obj1, obj2):
    x = obj1
    y = obj2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


# function to get x and y distances between 2 objects
def calc_distances(gameobject1, gameobject2):
    obj1, _ = gameobject1.get_coords()
    obj2, _ = gameobject2.get_coords()
    distx = obj2[0] - obj1[0]
    disty = obj2[1] - obj1[1]
    return distx, disty


# function for calculating euclidean distances between objects
def calc_euclidean_distance(gameobject1, gameobject2):
    obj1, _ = gameobject1.get_coords()
    obj2, _ = gameobject2.get_coords()
    dist = math.sqrt((obj2[1] - obj1[1])**2 + (obj2[0] - obj1[0])**2)
    return dist


# function to return velocity
def get_velocity(gameobject):
    obj, obj_past = gameobject.get_coords()
    vel = 0
    if obj_past is not None and not (obj[0] == obj_past[0] and obj[1] == obj_past[1]):
        vel = math.sqrt((obj_past[0] - obj[0])**2 + (obj_past[1] - obj[1])**2)
    return vel


# function to get dist to lin trajectory of one object
def get_lin_traj_distance(gameobject1, gameobject2):
    obj1, obj1_past = gameobject1.get_coords()
    obj2, obj2_past = gameobject2.get_coords()
    distx = 0
    disty = 0
    # if other object has moved
    if obj2_past is not None and not (obj2[0] == obj2_past[0] and obj2[1] == obj2_past[1]):
        # append trajectory cutting points
        m, c = get_lineq_param(obj2, obj2_past)
        # now calc target pos
        # y = mx + c substracted from its y pos
        disty = np.int16(m * obj1[0] + c) - obj1[1]
        # x = (y - c)/m substracted from its x pos
        distx = np.int16((obj1[1] - c) / m)  - obj1[0]
    return disty, distx


def convert_rgb_to_names(gameobject):
    rgb_tuple = gameobject.rgb
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS2_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    #print(gameobject.name, names[index])
    return f'closest match: {names[index]}', index


# just returns center of two game objects
def get_center(gameobject1, gameobject2):
    obj1, _ = gameobject1.get_coords()
    obj2, _ = gameobject2.get_coords()
    return (obj1[0] + obj2[0])/2, (obj1[1] + obj2[1])/2


# helper function to convert env info into custom list
# raw_features contains player x, y, ball x, y, oldplayer x, y, oldball x, y, etc...
# features are processed stuff for policy
def calc_preset_mifs(game_objects):
    features = []
    for i in game_objects:
        current_gameobject = game_objects[i]
        # append vel
        features.append(get_velocity(current_gameobject))
        _, color_index = convert_rgb_to_names(current_gameobject)
        features.append(color_index)
        # loop for axis distances
        for j in game_objects:
            # apped all manhattan distances to all other objects
            # which are not already calculated
            if j > i:
                # get and append distances
                distx, disty = calc_distances(current_gameobject, game_objects[j])
                features.append(distx) # append x dist
                features.append(disty) # append y dist
        # loop for euclidean distances
        for j in game_objects:
            # apped all euclidean distances to all other objects
            # which are not already calculated
            if j > i:
                # get and append distances
                features.append(calc_euclidean_distance(current_gameobject, game_objects[j]))
        # loop for trajectories
        for j in game_objects:
            # calculate movement paths of all other objects
            # and calculate distance to its x and y intersection
            if i != j:
                disty, distx = get_lin_traj_distance(current_gameobject, game_objects[j])
                features.append(disty)
                features.append(distx)
        # loop for centers
        for j in game_objects:
            # apped all axis centers to all other objects
            # which are not already calculated
            if j > i:
                # get and append distances
                cx, cy = get_center(current_gameobject, game_objects[j])
                features.append(cx) # append x dist
                features.append(cy) # append y dist
    #print(features)
    return features


# function getting list of functions and objects to calculate all from it
# objects: list with x and y and past x and y
# functions: TODO!!!!
def calc_given_mifs(objects, functions):
    i = 0
    features = []
    while i < len(objects):
        None
    return features






