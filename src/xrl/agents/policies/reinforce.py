import imp
from operator import mod
import gym
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm

from rtpt import RTPT

import xrl.utils.utils as xutils
from xrl.environments import env_manager
from xrl.agents.policies.policy_model import policy_net as Policy

from xrl.agents import Agent

PATH_TO_OUTPUTS = os.getcwd() + "/xrl/checkpoints/"
if not os.path.exists(PATH_TO_OUTPUTS):
    os.makedirs(PATH_TO_OUTPUTS)

model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_model.pth"


def select_action(features, policy, random_tr = -1, n_actions=3):
    input = torch.tensor(features).unsqueeze(0).float()
    probs = policy(input)
    #print(list(np.around(probs.detach().numpy(), 3)))
    sampler = Categorical(probs)
    action = sampler.sample()
    log_prob = sampler.log_prob(action)
    # select action when no random action should be selected
    if random.random() <= random_tr:
       action = random.randint(0, n_actions - 1)
    else:
        action = action.item()
    # return action and log prob
    return action, log_prob


def finish_episode(policy, optimizer, eps, cfg):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + cfg.train.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy, optimizer


# save model helper function
def save_policy(training_name, policy, episode, optimizer):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'policy': policy.state_dict(),
            'episode': episode,
            'optimizer': optimizer.state_dict()
            }, model_path)


# load model
def load_model(model_path, policy, optimizer=None):
    print("{} does exist, loading ... ".format(model_path))
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint['policy'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    i_episode = checkpoint['episode']
    return policy, optimizer, i_episode


def train(cfg, agent):
    print('Experiment name:', cfg.exp_name)
    torch.manual_seed(cfg.seed)
    print('Seed:', torch.initial_seed())
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)
    # init env to get params for policy net
    env = env_manager.make(cfg, True)
    n_actions = env.action_space.n
    gametype = xutils.get_gametype(env)
    _, ep_reward = env.reset(), 0
    obs, _, _, info = env.step(1)
    raw_features = agent.image_to_feature(obs, info, gametype)
    features = agent.feature_to_mf(raw_features)
    # init policy net
    print("Make hidden layer in nn:", cfg.train.make_hidden)
    policy = Policy(len(features), cfg.train.hidden_layer_size, n_actions, cfg.train.make_hidden)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.train.learning_rate)
    eps = np.finfo(np.float32).eps.item()
    i_episode = 1
    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        policy, optimizer, i_episode = load_model(model_path, policy, optimizer)
    print('Episodes:', cfg.train.num_episodes)
    print('Current episode:', i_episode)
    print("Random Action probability:", cfg.train.random_action_p)
    print('Max Steps per Episode:', cfg.train.max_steps)
    print('Gamma:', cfg.train.gamma)
    print('Learning rate:', cfg.train.learning_rate)
    # reinit agent with loaded policy model
    agent = Agent(f1=agent.feature_extractor, f2=agent.feature_to_mf, m=policy, f3=select_action)
    # setup last variables for init
    running_reward = None
    reward_buffer = 0
    # training loop
    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                    max_iterations=cfg.train.num_episodes)
    rtpt.start()
    pbar = tqdm(initial = i_episode, total = cfg.train.num_episodes)
    while i_episode < cfg.train.num_episodes:
        # init env
        _, ep_reward = env.reset(), 0
        # env loop
        t = 0
        while t < cfg.train.max_steps:  # Don't infinite loop while learning
            action, log_prob = agent.mf_to_action(features, agent.model, cfg.train.random_action_p, n_actions)
            policy.saved_log_probs.append(log_prob)
            # to env step
            _, reward, done, info = env.step(action)
            raw_features = agent.image_to_feature(obs, info, gametype)
            features = agent.feature_to_mf(raw_features)
            policy.rewards.append(reward)
            ep_reward += reward
            t += 1
            if done:
                break
        # only optimize when t < max ep steps
        if t >= cfg.train.max_steps:
            ep_reward = -25 #TODO: change to automatically game specific
        # replace first running reward with last reward for loaded models
        if running_reward is None:
            running_reward = ep_reward
        else:
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        reward_buffer += ep_reward
        policy, optimizer = finish_episode(policy, optimizer, eps, cfg)
        pbar.set_description('Last reward: {:.2f} | Running reward: {:.2f} | Steps: {} '.format(
            ep_reward, running_reward, t))
        #print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tSteps: {}       '.format(
        #    i_episode, ep_reward, running_reward, t), end="\r")
        if i_episode % cfg.train.log_steps == 0:
            avg_r = reward_buffer / cfg.train.log_steps
            writer.add_scalar('Train/Avg reward', avg_r, i_episode)
            reward_buffer = 0
        if (i_episode + 1) % cfg.train.save_every == 0:
            save_policy(cfg.exp_name, policy, i_episode + 1, optimizer)
        # finish episode
        i_episode += 1
        pbar.update(1)
        rtpt.step()


# eval function, returns trained model
def eval_load(cfg, agent):
    print('Experiment name:', cfg.exp_name)
    print('Evaluating Mode')
    print('Seed:', cfg.seed)
    print("Random Action probability:", cfg.train.random_action_p)
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)
    # init env
    env = env_manager.make(cfg)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, info = env.step(1)
    raw_features = agent.image_to_feature(obs, info, xutils.get_gametype(env))
    features = agent.feature_to_mf(raw_features)
    print("Make hidden layer in nn:", cfg.train.make_hidden)
    policy = Policy(len(features), cfg.train.hidden_layer_size, n_actions, cfg.train.make_hidden)
    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        policy, _, i_episode = load_model(model_path, policy)
        print('Episodes trained:', i_episode)
    policy.eval()
    # return policy 
    return policy
