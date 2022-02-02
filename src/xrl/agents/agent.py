# class of an agent of xmodrl
# each agent has 3 functions representing 
# the three steps in the pipeline


class Agent():
    def __init__(self, f1, f2, m = None, f3 = None):
        # choose the raw features extractor and set as first functions 
        #if cfg.raw_features_extractor == "atariari":
        #    self.image_to_feature = aari_raw_features_extractor
        self.image_to_feature = f1
        # choose the meaningful feature processing function and set as second functions
        self.feature_to_mf = f2
        # choose the meaningful feature processing function and set as second functions
        self.model = m
        self.mf_to_action = f3

        self.pipeline = [self.image_to_feature, 
                            self.feature_to_mf,
                            self.mf_to_action]

    def image_to_feature(self, images):
        return None

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
            if "__name__" in dir(feat):
                str_ret += f"\t-> {feat.__name__}\n"
            else:
                str_ret += f"\t-> {feat}\n"
        return str_ret
