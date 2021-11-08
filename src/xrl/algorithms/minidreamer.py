# own mini version of dreamer for processed features
import gym
import os
from matplotlib.pyplot import axis, step
import torch
import threading
import numpy as np

from atariari.benchmark.wrapper import AtariARIWrapper
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm
from torch.optim import Adam, SGD
from itertools import count
from rtpt import RTPT
from time import sleep, time

from xrl.algorithms.minidr.dataset import ModelDataset
from xrl.algorithms.minidr.model import WorldPredictor, Policy

import xrl.utils.utils as xutils

torch.autograd.set_detect_anomaly(True)

# world pred parameter
batch = 10
history_size = 50
min_history = 10
lr_pred = 1e-3
alpha = 1e-1   #alpha to scale down loss of state for loss of reward #old 1e-5

# policy parameter
lr_policy = 1e-3
gamma = 0.97

# misc parameter


# helper function to save all models
def save(
    path, 
    world_predictor, 
    policy,
    optim_predictor, 
    optim_policy,
    episode,
    global_step):
    # save model
    print("Saving {}".format(path))
    torch.save({
            'world_predictor': world_predictor.state_dict(),
            'policy': policy.state_dict(),
            'optim_predictor': optim_predictor.state_dict(),
            'optim_policy': optim_policy.state_dict(),
            'episode': episode,
            'global_step': global_step
            }, path)
    return None


# helper function to load all models
def load(path, 
    world_predictor, 
    policy,
    optim_predictor, 
    optim_policy):
    print("{} does exist, loading ... ".format(path))
    checkpoint = torch.load(path)
    world_predictor.load_state_dict(checkpoint['world_predictor'])
    policy.load_state_dict(checkpoint['policy'])
    if optim_predictor is not None:
        optim_predictor.load_state_dict(checkpoint['optim_predictor'])
    if optim_policy is not None:
        optim_policy.load_state_dict(checkpoint['optim_policy'])
    episode = checkpoint['episode']
    steps_done = checkpoint['global_step']
    return world_predictor, policy, optim_predictor, optim_policy, episode, steps_done


# function to select action
# TODO: fix for planning
def select_action(n_actions, policy):
    action = np.random.randint(0, n_actions)
    return action, None


def train(cfg):
    ### HYPERPARAMETERS ###
    print('Experiment name:', cfg.exp_name)
    torch.manual_seed(cfg.seed)
    print('Seed:', torch.initial_seed())
    print('Env name:', cfg.env_name)
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)


    # params for saving loading
    save_path = os.getcwd() + "/xrl/checkpoints/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + cfg.exp_name + "_model.pth"

    # device to use for training
    device = cfg.device

    # logger
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    # params for game
    env = AtariARIWrapper(gym.make(cfg.env_name))
    n_actions = env.action_space.n
    env.reset()
    raw_features, features, _, _ = xutils.do_step(env)
    len_raw_features = 12
    # init policy net
    
    ### MODELS ###
    predictor = WorldPredictor(input=len_raw_features).to(device)
    policy = Policy(len(features), cfg.train.hidden_layer_size, n_actions).to(device)
    criterion = torch.nn.MSELoss()
    optim_predictor = Adam(predictor.parameters(), lr=lr_pred)
    optim_policy = Adam(policy.parameters(), lr=lr_policy)

    history = []
    steps_done = [0]  
    i_episode = 1

    # load if checkpoint exists
    if os.path.isfile(save_path):
        predictor, policy, optim_predictor, optim_policy, i_episode, steps_done = load(save_path, predictor, policy, optim_predictor, optim_policy)
       
    # inner function to give it access to shared variables
    # loop to interactively fill history with newest episodes
    def gather_episode():
        with torch.no_grad():
            rtpt_s = RTPT(name_initials='DV', experiment_name=cfg.exp_name + "_interactive",
                    max_iterations=cfg.train.num_episodes)
            rtpt_s.start()
            while steps_done[0] < cfg.train.max_steps:
                obs = env.reset()
                raw_features, features, _, _ = xutils.do_step(env)
                episode = []
                done = False
                r_sum = 0
                t = 0
                while t < 50000:
                    a, _ = select_action(n_actions, policy)
                    # create episode entry with last state, action, state and reward
                    episode.append(raw_features)
                    episode.append(a)
                    raw_features, features, reward, done = xutils.do_step(env, a, raw_features)
                    # append reward to features to have both
                    episode.append(raw_features)
                    episode.append(reward)
                    r_sum += reward
                    #print("Step: {}, Reward: {}".format(t, rew), end="\r")
                    steps_done[0] += 1
                    if done:
                        break
                    t += 1
                writer.add_scalar('Train/Reward', r_sum, steps_done[0])
                history.append(episode)
                for _ in range(len(history) - history_size):
                    history.pop(0)
                rtpt_s.step()

    #start gathering episode thread
    t = threading.Thread(target=gather_episode)
    t.start()

    print("Dataset init")
    while len(history) < min_history:
        # check every second if first history entry is inside
        print("Wainting until history large enough (current size: {}, needed: {})".format(len(history), min_history), end="\r")
        sleep(1.0)
    print("done")

    ### DATASET ###
    ds = ModelDataset(history, history_size=history_size)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                    max_iterations=cfg.train.num_episodes)
    rtpt.start()
    pbar = tqdm(total=cfg.train.max_steps)
    pbar.update(n=steps_done[0])
    while steps_done[0] < cfg.train.max_steps:
        tmp_steps_done = steps_done[0]
        loss_pred = 0
        loss_s = 0
        loss_r = 0
        len_h = 0

        # create copy of predictor for policy training
        #policy_predictor = WorldPredictor(input=len_raw_features).to(device)
        #policy_predictor.load_state_dict(predictor.state_dict())

        train_s = None
        
        for ls, a, s, r in loader:
            ls = ls.to(device)
            a = a.unsqueeze(1).to(device)
            s = s.to(device)
            r = r.unsqueeze(1).to(device)

            ### TRAIN PREDICTOR
            
            # predict
            ps, pr = predictor(ls, a)

            #print(ls[0])
            #print(s[0])
            #print(ps[0])
            #print((ls[0]-ps[0]).detach().numpy())
            #print((s[0]-ps[0]).detach().numpy())
            #print(torch.mean(ls[0]-ps[0]))
            #print(torch.mean(s[0]-ps[0]))
            #print("###")
            #print(r)
            #print(pr)
            #print(ss)

            # calc loss
            optim_predictor.zero_grad()
            loss_s = criterion(ps, s)
            loss_r = criterion(pr, r)
            loss_pred = alpha * loss_s + loss_r
            loss_pred.backward()
            optim_predictor.step()

            ### TRAIN POLICY

            # TODO: planning

            if i_episode % 5 == 0:
                # log all
                writer.add_scalar('Train/Loss World Predictor', loss_pred, steps_done[0])
                writer.add_scalar('Train/Loss World Predictor State', loss_s, steps_done[0])
                writer.add_scalar('Train/Loss World Predictor Reward', loss_r, steps_done[0])
                writer.add_scalar('Train/Diff State Identity', torch.mean(ls-ps), steps_done[0])
                writer.add_scalar('Train/Diff State Next State', torch.mean(s-ps), steps_done[0])
                # set logging variables
                len_h = len(history)
                # display
                pbar.set_postfix(
                    l_pred = loss_pred.item(),
                    len_h = len_h,
                    i_episode = i_episode
                )

        # update tqdm
        pbar.update(n = steps_done[0] - tmp_steps_done)

        # save
        if i_episode % cfg.train.save_every == 0:
            save(save_path, predictor, policy, optim_predictor, optim_policy, i_episode, steps_done)

        # end episode
        i_episode += 1
        rtpt.step()