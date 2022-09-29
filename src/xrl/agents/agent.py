# class of an agent of xmodrl
# each agent has 3 functions representing 
# the three steps in the pipeline

from cProfile import label
import numpy as np

from xrl.agents.image_extractors.label_encoder import extract_from_labels


class Agent():
    def __init__(self, f1, f2, m = None, f3 = None):
        # choose the raw features extractor and set as first functions 
        # function to extract raw features
        self.feature_extractor = f1
        self.raw_features = []
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
        #print(labels)
        # encode given labels dict
        new_raw_features = extract_from_labels(labels, gametype)
        #print("extracted raw features:", new_raw_features)
        if len(self.raw_features) < 1:
            # init with double length
            self.raw_features = [None] * (len(new_raw_features)) * 2
        else:
            # roll to have the old ones at front for overwriting
            self.raw_features = np.roll(self.raw_features, len(new_raw_features))
        # now add first len(new_raw_features) values
        self.raw_features[0:len(new_raw_features)] = new_raw_features[0:len(new_raw_features)]
        return self.raw_features

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
