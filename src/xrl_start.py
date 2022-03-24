# main file for all rl algos

import random
from sklearn import tree
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from captum.attr import IntegratedGradients
from torch.distributions import Categorical
from torchinfo import summary
from termcolor import colored
from tqdm import tqdm
from rtpt import RTPT

from xrl.agents.policies import reinforce
from xrl.agents.policies import genetic_rl as genetic
from xrl.agents.policies import dreamer_v2
from xrl.agents.policies import minidreamer

#import xrl.utils.plotter as xplt
import xrl.utils.video_logger as vlogger
import xrl.utils.utils as xutils
import xrl.utils.plotter as xplt
import xrl.utils.tree_explainer as tx

# otherwise genetic loading model doesnt work, torch bug?
from xrl.agents.policies.policy_model import policy_net
from xrl.environments import env_manager

# all extractor and processor to later select with infos from config file
# feature extractor functions
from xrl.agents.image_extractors.aari_raw_features_extractor import get_labels
from xrl.agents.image_extractors.color_extractor import ColorExtractor
from xrl.agents.image_extractors.interactive_color_extractor import IColorExtractor

# feature processing functions
from xrl.agents.feature_processing.aari_feature_processer import calc_preset_mifs
# agent class
from xrl.agents import Agent
import xrl.utils.pruner as pruner


# helper function to select action from loaded agent
# has random probability parameter to test stability of agents
# function to select action by given features
# TODO: Remove like genetic algo
def select_action(features, policy, random_tr = -1, n_actions=3):
    sample = random.random()
    if sample > random_tr:
        # calculate probabilities of taking each action
        probs = policy(torch.tensor(features).unsqueeze(0).float())
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample()
    else:
        action = random.randint(0, n_actions - 1)
    return action


# function to test agent loaded via main switch
def play_agent(agent, cfg):
    # init env
    env = env_manager.make(cfg, True)
    n_actions = env._env.action_space.n
    gametype = xutils.get_gametype(env)
    _, ep_reward = env.reset(), 0
    obs, _, _, _, info = env.step(1)
    raw_features = agent.image_to_feature(obs, info, gametype)
    features = agent.feature_to_mf(raw_features)
    # init objects
    summary(agent.model, input_size=(1, len(features)), device=cfg.device)
    # make multiple runs for eval
    runs = 5
    print("Runs:", runs)
    rewards = []
    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name + "_EVAL",
                max_iterations=runs)
    rtpt.start()
    for run in tqdm(range(runs)):
        # env loop
        t = 0
        ep_reward = 0
        env.reset()
        while t < cfg.train.max_steps:  # Don't infinite loop while playing
            action = agent.mf_to_action(features, agent.model, -1, n_actions)
            features = torch.tensor(features).unsqueeze(0).float()
            if cfg.liveplot:
                plt.imshow(obs, interpolation='none')
                plt.plot()
                plt.pause(0.001)  # pause a bit so that plots are updated
                plt.clf()
            #print('Reward: {:.2f}\t Step: {:.2f}'.format(
            #        ep_reward, t), end="\r")
            obs, reward, done, done2, info = env.step(action)
            raw_features = agent.image_to_feature(obs, info, gametype)
            features = agent.feature_to_mf(raw_features)
            ep_reward += reward
            t += 1
            if done or done2:
                break
        rewards.append(ep_reward)
        #print('Final reward: {:.2f}\tSteps: {}'.format(ep_reward, t))
        rtpt.step()
    print(rewards)
    print("Mean of Rewards:", sum(rewards) / len(rewards))



# function to call reinforce algorithm
def use_reinforce(cfg, mode, agent):
    print("Selected algorithm: REINFORCE")
    if mode == "train":
        reinforce.train(cfg, agent)
    else:
        policy = reinforce.eval_load(cfg, agent)
        # reinit agent with loaded model and eval function
        agent = Agent(f1=agent.feature_extractor, f2=agent.feature_to_mf, m=policy, f3=select_action)
        if mode == "eval":
            play_agent(agent=agent, cfg=cfg)
        elif mode == "explain":
            explain(agent=agent, cfg=cfg)


# function to call deep neuroevolution algorithm
def use_genetic(cfg, mode, agent):
    print("Selected algorithm: Deep Neuroevolution")
    if mode == "train":
        genetic.train(cfg, agent)
    else:
        agent = genetic.eval_load(cfg, agent)
        if mode == "eval":
            play_agent(agent=agent, cfg=cfg)
        elif mode == "explain":
            explain(agent=agent, cfg=cfg)


# function to call dreamerv2
def use_dreamerv2(cfg, mode):
    print("Selected algorithm: DreamerV2")
    print("Implementation has errors, terminating ...")
    #if cfg.mode == "train":
    #    dreamer_v2.train(cfg)
    #elif cfg.mode == "eval":
    #    dreamer_v2.eval(cfg)


# function to call minidreamer
def use_minidreamer(cfg, mode):
    print("Selected algorithm: Minidreamer")
    if mode == "train":
        minidreamer.train(cfg)
    elif mode == "eval":
        print("Eval not implemented ...")


# init agent function
def init_agent(cfg):
    focus_mode = cfg.focus_mode
    print("Focus mode:", focus_mode)
    # init correct raw features extractor
    rfe = None
    if cfg.raw_features_extractor == "atariari":
        print("Raw Features Extractor:", "atariari")
        rfe = get_labels
    elif cfg.raw_features_extractor == "CE":
        print("Raw Features Extractor:", "ColorExtractor")
        game = cfg.env_name.replace("Deterministic", "").replace("-v4", "")
        rfe = ColorExtractor(game=game, load=False)
    # set correct mifs
    # and set agent
    if focus_mode == "scobot":
        None
    elif focus_mode == "iscobot":
        print("Not implemented, sorry :(")
        exit(1)
    elif focus_mode == "iscobot-preset":
        return Agent(f1=rfe, f2=calc_preset_mifs)
    else:
        print("Unknown mode, terminating...")
        exit(1)
    


# main function
# switch for each algo
def xrl(cfg, mode):
    # init agent without third part of pipeline
    agent = init_agent(cfg)
    # algo selection
    # 1: REINFORCE
    # 2: Deep Neuroevolution
    # 3: DreamerV2
    # 4: Minidreamer
    if cfg.rl_algo == 1:
        use_reinforce(cfg, mode, agent)
    elif cfg.rl_algo == 2:
        use_genetic(cfg, mode, agent)
    elif cfg.rl_algo == 3:
        raise RuntimeError("Pls don't use, not working :( ...")
        use_dreamerv2(cfg, mode)
    elif cfg.rl_algo == 4:
        raise RuntimeError("Pls don't use, Minidreamer not finished :( ...")
        use_minidreamer(cfg, mode)
    else:
        print("Unknown algorithm selected")
