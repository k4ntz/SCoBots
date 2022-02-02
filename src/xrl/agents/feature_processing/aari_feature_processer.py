# file for raw feature processing to meaningful features
# with atari ari gym environments

import numpy as np
import math


# helper function to calc linear equation
def get_lineq_param(obj1, obj2):
    x = obj1
    y = obj2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


# helper function to convert env info into custom list
# raw_features contains player x, y, ball x, y, oldplayer x, y, oldball x, y, etc...
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
    return features