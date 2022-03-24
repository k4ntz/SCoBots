import imp
from operator import mod
import gym
import numpy as np
import os
import random
import math 
import csv

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

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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
    returns = torch.tensor(returns, device=dev)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum() #try mean
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy, optimizer, policy_loss


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


def calc_fr(features):
        if features is None:
            return 0
        p_coords = features[0:2]
        e_coords = features[2:4]
        b_coords = features[4:6]
        p_b_distance_now = math.sqrt((p_coords[0] - b_coords[0])**2 + (p_coords[1] - b_coords[1])**2)
        # if not raw_features[3] is None:
        #    p_b_distance_past = math.sqrt((p_coords_past[0] - b_coords_past[0])**2 + (p_coords_past[1] - b_coords_past[1])**2)
        # else:
        #    p_b_distance_past = 9000 #random high number for first step
        # features.append(p_b_distance_now)
        # features.append(p_b_distance_past)
        return p_b_distance_now


def train(cfg, agent):
    with open(os.getcwd() + cfg.logdir + cfg.exp_name+'.csv', 'w+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["episode", "steps", "natural_reward", "feedback_reward", "combined_reward", "feedback_alpha"])
    print('Experiment name:', cfg.exp_name)
    torch.manual_seed(cfg.seed)
    print('Seed:', torch.initial_seed())
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
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

    # load saved model if exists
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
    cr_buffer = 0
    nr_buffer = 0
    nr_history = []
    cr_history = []
    max_distance_observed = 1 #change to a generic approach

    feedback_alpha = cfg.train.feedback_alpha
    delta = cfg.train.feedback_delta
    sign = -1

    # training loop
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name,
                    max_iterations=cfg.train.num_episodes)
    rtpt.start()
    while i_episode < cfg.train.num_episodes: #TODO: name to trajectories
        # init env
        _, ep_comb_reward, ep_natural_reward, ep_feedback_reward = env.reset(), 0, 0, 0
        # env loop
        t = 0
        last_raw_features = None
        last_features = None
        while t < cfg.train.max_steps:  # Don't infinite loop while learning TODO: naming
            action, log_prob = agent.mf_to_action(features, agent.model, cfg.train.random_action_p, n_actions)
            policy.saved_log_probs.append(log_prob)
            _, natural_reward, done, info = env.step(action)
            # reward <- distance(player, ball)
            # reward_list.append((reward_distance, 0.7)) in the future probably
            raw_features = agent.image_to_feature(info, last_raw_features, gametype) #TODO doublecheck para order
            features = agent.feature_to_mf(raw_features)

            # distance delta of player<->ball between present and past
            b_p_distance_now = calc_fr(features)
            b_p_distance_past = calc_fr(last_features)
            if b_p_distance_now > max_distance_observed:
                max_distance_observed = b_p_distance_now

            # difference of potentials scale to max_distance
            feedback_reward = (-b_p_distance_now - -b_p_distance_past ) / max_distance_observed
            comb_reward = natural_reward + (feedback_alpha * feedback_reward)

            # normalize to [0,1]. convex feedback_reward handling
            # feedback_reward = 1 -  (b_p_distance_now / max_distance_observed )
            # alpha = feedback_alpha
            # comb_reward = (1 - alpha) * natural_reward + alpha * feedback_reward
            
            # difference of potentials normalized with max_distance_observed
            # feedback_reward = (- b_p_distance_now + b_p_distance_past) / max_distance_observed # -distance - -distance
            # comb_reward = reward + feedback_reward

            # distance_delta = features[-2] - features[-1] 
            # reward distance reduction
            #if distance_delta < 0:
            #    feedback_reward = 1
            #elif distance_delta > 0:
            #    feedback_reward = -1
            #else:
            #    feedback_reward = 0
            # reward weights
            #comb_reward = float(feedback_reward)

            policy.rewards.append(comb_reward)
            ep_comb_reward += comb_reward
            ep_natural_reward += natural_reward
            ep_feedback_reward += feedback_reward
            last_raw_features = raw_features
            last_features = features
            t += 1
            if done:
                break

        # S: not used for optimization, just for logging
        # only optimize when t < max ep steps
        if t >= cfg.train.max_steps:
            ep_comb_reward = -21 #TODO: change to automatically game specific
        # replace first running reward with last reward for loaded models

        # S:  this reward is just for tracking purposes, not used for training
        if running_reward is None:
            running_reward = ep_comb_reward
        else:
            running_reward = 0.05 * ep_comb_reward + (1 - 0.05) * running_reward

        cr_buffer += ep_comb_reward
        nr_buffer += ep_natural_reward

        policy, optimizer, loss = finish_episode(policy, optimizer, eps, cfg)
        print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tNR: {:.2f}\tSteps: {}       '.format(
            i_episode, ep_comb_reward, running_reward, ep_natural_reward, t)) #, end="\r")
        
        # turn feedback reward alpha up or down when natural reward is stuck
        # TODO: make reward histories persistent to be able to continue training after aborting
        cr_history.append(ep_comb_reward)
        nr_history.append(ep_natural_reward)
        first_probe_ep = cfg.train.stale_window + cfg.train.stale_probe_window
        if i_episode >= first_probe_ep and len(nr_history) >= first_probe_ep:
            if (i_episode % cfg.train.stale_window == cfg.train.stale_probe_window):
                # cr_past = cr_history[-cfg.train.stale_window]
                present_probe_start = - cfg.train.stale_probe_window
                present_probe_end = -1
                present_window =  nr_history[present_probe_start : present_probe_end]
                present_avg = sum(present_window) / len(present_window)
                past_probe_start = -cfg.train.stale_window - cfg.train.stale_probe_window
                past_probe_end = -cfg.train.stale_window 
                past_window = nr_history[past_probe_start : past_probe_end]
                #print(past_probe_start)
                #print(past_probe_end)
                #print(past_window)
                past_avg = sum(past_window) / len(past_window)
                change = (present_avg - past_avg) / abs(past_avg)
                #print("")
                #print(str(past_avg) + " " + str(present_avg) + " " + str(change))

                #print("")
                if feedback_alpha == 0:
                    sign = 1
                elif feedback_alpha == 1.0:
                    sign = -1


                if change < cfg.train.stale_threshold:
                    feedback_alpha+= sign * delta
                    #print(sign)

                #print("")

        # TODO: handle pausing and resuming training
        # fine grained csv logging
        with open(os.getcwd() + cfg.logdir + cfg.exp_name+'.csv', 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([i_episode, t, ep_natural_reward, ep_feedback_reward, ep_comb_reward, feedback_alpha])
    
        # tfboard logging
        if i_episode % cfg.train.log_steps == 0:
            avg_r = cr_buffer / cfg.train.log_steps
            avg_nr = nr_buffer / cfg.train.log_steps
            writer.add_scalar('rewards/avg natural', avg_nr, i_episode)
            writer.add_scalar('rewards/avg combined', avg_r, i_episode)
            writer.add_scalar('params/feedback_alpha', feedback_alpha, i_episode)
            cr_buffer = 0
            nr_buffer = 0
      
        # checkpointing
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
    model_path = model_name(cfg.exp_name + "-seed" + str(cfg.seed))
    if os.path.isfile(model_path):
        policy, _, i_episode = load_model(model_path, policy)
        print('Episodes trained:', i_episode)
    policy.eval()
    # return policy 
    return policy
