import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import matplotlib.pyplot as plt
import numpy as np

try:
    from remix.rules.ruleset import Ruleset
    from remix.rules.rule import RulePredictMechanism
except:
    print("Could not import remix")
from tqdm import tqdm
from rtpt import RTPT
from algos import reinforce
from algos import genetic_rl as genetic
from scobi import Environment
from experiments.utils.normalizer import Normalizer
from pathlib import Path
import matplotlib as mpl
import json
import os
import math
from experiments.utils.xrl_utils import Drawer
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning) 

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_actions_for_states(cfg, states):
    # initialize output file
    print("eclaire_dir: ", cfg.eclaire.eclaire_dir)
    outfile_path_actions = os.path.join("..", cfg.eclaire.eclaire_dir, "actions_eclaire.npy")
    ruleset_path = os.path.join("..", cfg.eclaire.eclaire_dir, "output.rules")
    ruleset = Ruleset().from_file(ruleset_path)

    features = states
    actions, rules, scores = ruleset.predict_and_explain(features,
                                                        only_positive=True,
                                                        use_confidence=True,
                                                        aggregator=RulePredictMechanism.Max)
    actions = [int(action) for action in actions]
    np.save(outfile_path_actions, actions)

def explain_agent(cfg, normalizer, ruleset_path, env=None):
    runs = 5
    # init env
    draw = False
    if env is None:
        env = Environment(cfg.env_name,
                          cfg.seed,
                          reward=cfg.scobi_reward_shaping,
                          hide_properties=cfg.scobi_hide_properties,
                          focus_dir=cfg.scobi_focus_dir,
                          focus_file=cfg.scobi_focus_file,
                          draw_features=draw)

    _, ep_reward = env.reset(), 0
    obs, _, _, _, info = env.step(1)
    obs_raw = env.original_obs
    features = obs
    print("Runs:", runs)
    rewards = []
    all_sco_rewards = []
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name + "_EVAL", max_iterations=runs)
    rtpt.start()
    
    drawer = Drawer(obs_raw)

    # initialize denorm dict
    denorm_dict = None
    if normalizer is not None:
        denorm_dict = {}
        fnames = env.focus.get_vector_entry_descriptions()
        fnames = [f.replace(" ", "") for f in fnames]
        #norm_state = normalizer.get_state()
        #assert(len(norm_state) == len(fnames))
        #for state, name in zip(norm_state, fnames):
        #    mean = state["m"]
        #    variance = state["s"] / (state["n"])
        #    standard_deviation = math.sqrt(variance)
        #    denorm_dict[name] = (mean, standard_deviation)
        mean = normalizer.mean
        variance = normalizer.var
        for m, var, name in zip(mean, variance, fnames):
            standard_deviation = math.sqrt(var)
            denorm_dict[name] = (m, standard_deviation)

    ruleset = Ruleset().from_file(ruleset_path)

    # initialize output file
    if "eclaire" in cfg:
        print("eclaire_dir: ", cfg.eclaire.eclaire_dir)
        outfile_path = os.path.join("..", cfg.eclaire.eclaire_dir, "remix_policy_rewards.csv")

    for run in tqdm(range(runs)):
        t = 0
        ep_reward = 0
        sco_reward = 0
        env.reset()
        # use tqdm to show progress bar
        tqdm.write("Run: " + str(run))
        bar = tqdm(total=cfg.train.max_steps_per_trajectory)
        while t < cfg.train.max_steps_per_trajectory:
            if not drawer.pause:
                features = normalizer.normalize(features)
                my_tuple = ruleset.predict_and_explain(features,
                                                    only_positive=True,
                                                    use_confidence=True,
                                                    aggregator=RulePredictMechanism.Max)
                if len(my_tuple) == 1: # for some enum values in RulePredictMechanism
                    action, rules, scores = my_tuple[0][0], my_tuple[0][1], my_tuple[0][2] 
                else: # for other enum values in RulePredictMechanism e.g. Max
                    action, rules, scores = my_tuple[0], my_tuple[1], my_tuple[2]
                action = int(action[0])

                if draw:
                    drawer.draw_explain(rules[0][0], action, env, normalize = normalizer is not None, denorm_dict = denorm_dict)
                obs, scobi_reward, done, done2, info = env.step(action)
                obs_raw = env.original_obs
                features = obs
                ep_reward += env.original_reward
                sco_reward += scobi_reward
                if env.original_reward != 0:
                    print("ep_reward:", ep_reward)
                t += 1
                bar.update(1)
                if done or done2:
                    break
            if draw:
                drawer.update(run, t)
        rewards.append(ep_reward)
        all_sco_rewards.append(sco_reward)
        rtpt.step()
    
    
    # save rewards
    if "eclaire" in cfg:
        # only save if outputfile_path does not exist
        if not os.path.exists(outfile_path):
            pd.DataFrame(rewards).to_csv(outfile_path, index=False)

    print(rewards)
    print(all_sco_rewards)
    print("Mean of Env Rewards:", sum(rewards) / len(rewards))
