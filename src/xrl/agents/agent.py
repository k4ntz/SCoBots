# class of an agent of xmodrl
# each agent has 3 functions representing 
# the three steps in the pipeline

from cProfile import label
import numpy as np

from xrl.agents.image_extractors.label_encoder import extract_from_labels
from xrl.agents.game_object import GameObject
from xrl.agents.image_extractors.rgb_extractor import extract_rgb_value


class Agent():
    def __init__(self, f1, f2, m = None, f3 = None):
        # choose the raw features extractor and set as first functions 
        # function to extract raw features
        self.feature_extractor = f1
        self.game_objects = {}
        # choose the meaningful feature processing function and set as second functions
        self.feature_to_mf = f2
        # choose the meaningful feature processing function and set as second functions
        self.model = m
        self.mf_to_action = f3

        self.pipeline = [self.image_to_feature, 
                            self.feature_to_mf,
                            self.mf_to_action]

    # main function to extract features from image
    def feature_extractor(image, gametype):
        return None
        
    # to have the last used features, there is a wrapper and temp variable
    # for the raw features from the last frame
    def image_to_feature(self, images, info, gametype):
        gameobject_info = self.feature_extractor(images, info, gametype)
        # encode given labels dict and its names
        #new_raw_features, gameobject_names = extract_from_labels(labels, gametype)
        # initial generate game objects when not in dict
        for key in gameobject_info:
            if not (key in self.game_objects):
                tmp = gameobject_info[key]
                # add rgb and wh inital to game object since its static
                rgb = [tmp[4], tmp[5], tmp[6]]
                wh = [tmp[2], tmp[3]]
                self.game_objects[key] = GameObject(key, rgb, wh)
        # add coordinates and color
        for key in self.game_objects:
            self.game_objects[key].update_coords(gameobject_info[key][0],gameobject_info[key][1])
        return self.game_objects

    def feature_to_mf(self, feature):
        return None

    def mf_to_action(self, mf_feature):
        return None

    def __call__(self, x):
        results = []
        for func in self.pipeline:
            x = func(x)
            results.append(x)
        return results

    def choose(self):
        pass

    def __repr__(self):
        str_ret = "Pipeline Agent consisting of:\n"
        for feat in self.pipeline:
            if feat is not None:
                str_ret += f"\t-> {feat.__name__}\n"
            else:
                str_ret += f"\t-> {feat}\n"
        return str_ret
