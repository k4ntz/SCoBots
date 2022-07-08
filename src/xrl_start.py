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
from xrl.agents.feature_processing.aari_feature_processer import preprocess_raw_features
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
    # return action
    return action


# function to test agent loaded via main switch
def play_agent(agent, cfg):
    # init env
    env = env_manager.make(cfg, True)
    n_actions = env.action_space.n
    gametype = xutils.get_gametype(env)
    _, ep_reward = env.reset(), 0
    obs, _, _, info = env.step(1)
    raw_features = agent.image_to_feature(obs, info, gametype)
    features = agent.feature_to_mf(raw_features)
    # init objects
    summary(agent.model, input_size=(1, len(features)), device=cfg.device)
    logger = vlogger.VideoLogger(size=(480, 480))
    ig = IntegratedGradients(agent.model)
    ig_sum = []
    ig_action_sum = []
    feature_titles = xplt.get_feature_titles(int(len(raw_features)/2))
    # env loop
    plotter = xplt.Plotter()
    t = 0
    # env.reset()
    max_value = 0
    while t < 3000:  # Don't infinite loop while playing
        action = agent.mf_to_action(features, agent.model, cfg.train.random_action_p, n_actions)
        features = torch.tensor(features).unsqueeze(0).float()
        if cfg.make_video:
            img = plotter.plot_IG_img(ig, cfg.exp_name, features, feature_titles, action, obs, cfg.liveplot)
            logger.fill_video_buffer(img)
        elif cfg.liveplot:
            plt.imshow(obs, interpolation='none')
            plt.plot()
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.clf()
        else:
            #ig_sum.append(xplt.get_integrated_gradients(ig, features, action))
            #ig_action_sum.append(np.append(xplt.get_integrated_gradients(ig, features, action), [action]))
            None
        print('Reward: {:.2f}\t Step: {:.2f}'.format(
                ep_reward, t), end="\r")
        obs, reward, done, info = env.step(action)
        raw_features = agent.image_to_feature(obs, info, gametype)
        features = agent.feature_to_mf(raw_features)
        ep_reward += reward
        t += 1
        if done:
            print("\n")
            break
    print('Final reward: {:.2f}\tSteps: {}'.format(ep_reward, t))
    print(max_value)
    if cfg.make_video:
        logger.save_video(cfg.exp_name)
    #elif not cfg.liveplot:
    #    ig_sum = np.asarray(ig_sum)
    #    ig_action_sum = np.asarray(ig_action_sum)
    #    ig_mean = np.mean(ig_sum, axis=0)
    #    # create dict with feature as key and ig-mean als value
    #    zip_iterator = zip(feature_titles, ig_mean)
    #    ig_dict = dict(zip_iterator)
    #    for k in ig_dict:
    #        print(k + ": " + str(ig_dict[k]))


# function for experimental tests, do not use!
def explain(agent, cfg):
    print(colored("CREATING EXPLAINING DECISION TREE", "green"))
    # init env
    env = env_manager.make(cfg, True)
    n_actions = env.action_space.n
    gametype = xutils.get_gametype(env)
    # TODO: Fix, because 100% of avg weight does not make any sense lol
    #agent.model, _ = pruner.prune_nn(agent.model, "avg-pr", 0.7, len(features))
    # data collecting loop
    x = []
    y = []
    data_collecting_loops = 5
    for i in tqdm(range(data_collecting_loops)):
        _, ep_reward = env.reset(), 0
        obs, _, _, info = env.step(1)
        raw_features = agent.image_to_feature(obs, info, gametype)
        features = agent.feature_to_mf(raw_features)
        t = 0
        action_randomness = random.random()
        while t < 3000:  # Don't infinite loop while playing
            # get maybe random action
            action = agent.mf_to_action(features, agent.model, action_randomness, n_actions)
            # get the actual action decided by policy without randomness in current situation
            true_action = agent.mf_to_action(features, agent.model, -1, n_actions)
            # save feature and action
            x.append(features)
            y.append(float(true_action))
            features = torch.tensor(features).unsqueeze(0).float()
            #print('Reward: {:.2f}\t Step: {:.2f}'.format(ep_reward, t), end="\r")
            # do action
            obs, reward, done, info = env.step(action)
            raw_features = agent.image_to_feature(obs, info, gametype)
            features = agent.feature_to_mf(raw_features)
            ep_reward += reward
            t += 1
            if done:
                #print("\n")
                break
        #print('Final reward: {:.2f}\tSteps: {}'.format(ep_reward, t))
    # test explainable conversion stuff
    feature_titles = xplt.get_feature_titles(int(len(raw_features) / 2))
    action_names_full = env.env.get_action_meanings()
    action_n_list = [int(i) for i in set(y)]
    action_names = [action_names_full[i] for i in action_n_list]
    tree_explainer = tx.TreeExplainer(x, y, feature_titles, action_names)
    tree_explainer.train()
    tree_explainer.visualize()
    # now play the game with the decision tree
    print("Now playing the game with the DT ...")
    _, ep_reward = env.reset(), 0
    obs, _, _, info = env.step(1)
    raw_features = agent.image_to_feature(obs, info, gametype)
    features = agent.feature_to_mf(raw_features)
    t = 0
    while t < 3000:  # Don't infinite loop while playing
        f = []
        f.append(features)
        feature_row = pd.DataFrame(f[:], columns=feature_titles)
        action = int(tree_explainer.tree.predict(feature_row)[0])
        print('Reward: {:.2f}\t Step: {:.2f}'.format(ep_reward, t), end="\r")
        # plot 
        if False:
            plt.imshow(obs, interpolation='none')
            plt.plot()
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.clf()
        # do action
        obs, reward, done, info = env.step(action)
        raw_features = agent.image_to_feature(obs, info, gametype)
        features = agent.feature_to_mf(raw_features)
        ep_reward += reward
        t += 1
        if done:
            #print("\n")
            break
    print('\n\nFinal reward: {:.2f}\tSteps: {}'.format(ep_reward, t))



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
    # init correct raw features extractor
    rfe = None
    if cfg.raw_features_extractor == "atariari":
        print("Raw Features Extractor:", "atariari")
        rfe = get_labels
    elif cfg.raw_features_extractor == "CE":
        print("Raw Features Extractor:", "ColorExtractor")
        game = cfg.env_name.replace("Deterministic", "").replace("-v4", "")
        rfe = ColorExtractor(game=game, load=False)
    # create agent and return
    return Agent(f1=rfe, f2=preprocess_raw_features)


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
