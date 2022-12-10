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
        self.game_objects = []
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
        labels = self.feature_extractor(images, info, gametype)
        # encode given labels dict and its names
        new_raw_features, gameobject_names = extract_from_labels(labels, gametype)
        # initial generate game objects
        if len(self.game_objects) < 1:
            for name in gameobject_names:
                self.game_objects.append(GameObject(name))
        # add coordinates and color
        for i in range(len(self.game_objects)):
            tmp_raw_feature = new_raw_features[i]
            self.game_objects[i].update_coords(tmp_raw_feature[0], tmp_raw_feature[1])
            self.game_objects[i].rgb = extract_rgb_value(images, tmp_raw_feature, gametype)
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
