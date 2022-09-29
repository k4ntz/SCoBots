#############
# DELETE ME #
#############

# TODO: Finish refactoring, so that this file is no longer needed!!!!


# main file which is selecting the raw features extractor 
# and the processer to meaninful features

import xrl.agents.image_extractors.aari_raw_features_extractor as aari_raw_features_extractor
import xrl.agents.feature_processing.aari_feature_processer as aari_features_processer


# helper function to get features
def do_step(env, action=1, last_raw_features=None, raw_features_processer="atariari"):
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
    raw_features = aari_raw_features_extractor.get_raw_features(info, last_raw_features, gametype=gametype)
    raw_features, features = aari_features_processer.preprocess_raw_features(raw_features)
    raise RuntimeError("Looks like refactoring is not complete for this call :( ...")
    return raw_features, features, reward, done
