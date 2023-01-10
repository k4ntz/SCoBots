# main file for all rl algos

import random
import torch
import matplotlib.pyplot as plt

from torch.distributions import Categorical
from torchinfo import summary
from tqdm import tqdm
from rtpt import RTPT

from algos import reinforce
from algos import genetic_rl as genetic
from scobi import Environment

#import xrl.utils.plotter as xplt
import utils.video_logger as vlogger
import utils.utils as xutils
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# agent class
#from agent import Agent

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# helper function to select action from loaded agent
# has random probability parameter to test stability of agents
# function to select action by given features
# TODO: Remove like genetic algo
def select_action(features, policy, random_tr = -1, n_actions=3):
    sample = random.random()
    if sample > random_tr:
        # calculate probabilities of taking each action
        f = torch.tensor(features).unsqueeze(0).float().to(dev)
        probs = policy(f)
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample()
    else:
        action = random.randint(0, n_actions - 1)
    return action

def re_select_action(*args, **kwargs):
    return reinforce.select_action(*args, **kwargs)[0]



# function to test agent loaded via main switch
def play_agent(cfg, model, select_action_func):
    # init env

    env = Environment(cfg.env_name, interactive=cfg.scobi_interactive, focus_dir=cfg.scobi_focus_dir, focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    #gametype = xutils.get_gametype(env)
    _, ep_reward = env.reset(), 0
    obs, _, _, _, info = env.step(1)
    features = obs

    # init objects
    summary(model, input_size=(1, len(features)), device=cfg.device)
    # make multiple runs for eval
    runs = 10
    print("Runs:", runs)
    rewards = []
    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name + "_EVAL", max_iterations=runs)
    rtpt.start()
    model.to(dev)
    for run in tqdm(range(runs)):
        # env loop
        t = 0
        ep_reward = 0
        env.reset()
        while t < cfg.train.max_steps:  # Don't infinite loop while playing
            #features = torch.tensor(features).unsqueeze(0).float()
            action = select_action_func(features, model, -1, n_actions)
            if cfg.liveplot:
                plt.imshow(obs, interpolation='none')
                plt.plot()
                plt.pause(0.001)  # pause a bit so that plots are updated
                plt.clf()
            #print('Reward: {:.2f}\t Step: {:.2f}'.format(
            #        ep_reward, t), end="\r")
            obs, reward, done, done2, info = env.step(action)
            features = obs
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
def use_reinforce(cfg, mode):
    if mode == "train":
        reinforce.train(cfg)
    else:
        if mode == "eval":
            model = reinforce.eval_load(cfg)
            play_agent(cfg, model, re_select_action)
        elif mode == "explain":
            pass
            # explain(agent=agent, cfg=cfg)


# function to call deep neuroevolution algorithm
def use_genetic(cfg, mode):
    print("Selected algorithm: Deep Neuroevolution")
    if mode == "train":
        genetic.train(cfg)
    else:
        agent = genetic.eval_load(cfg, agent)
        if mode == "eval":
            play_agent(agent=agent, cfg=cfg)
        elif mode == "explain":
            pass
            # explain(agent=agent, cfg=cfg)


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
    if focus_mode in ["scobot", "iscobot"]:
        dummy_agent = Agent(f1=rfe, f2=calc_preset_mifs)
        env = env_manager.make(cfg, True)
        #n_actions = env.action_space.n
        actions = env._env.unwrapped.get_action_meanings()
        gametype = xutils.get_gametype(env)
        _, _ = env.reset(), 0
        obs, _, _, _, info = env.step(1)
        raw_features = dummy_agent.image_to_feature(obs, info, gametype)
        focus = Focus(cfg, raw_features, actions)
        return Agent(f1=rfe, f2=focus.get_feature_vector)
    elif focus_mode == "iscobot-preset":
        return Agent(f1=rfe, f2=calc_preset_mifs)
    else:
        print("Unknown mode, terminating...")
        exit(1)
    


# main function
# switch for each algo
def xrl(cfg, mode):
    # algo selection
    # 1: REINFORCE
    # 2: Deep Neuroevolution
    if cfg.rl_algo == 1:
        use_reinforce(cfg, mode)
    elif cfg.rl_algo == 2:
        use_genetic(cfg, mode)
    else:
        print("Unknown algorithm selected")
