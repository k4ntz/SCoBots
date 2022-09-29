# from: https://github.com/vincent-thevenin/DreamerV2-Pytorch
# original paper: https://arxiv.org/pdf/2010.02193.pdf

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import threading
import torch
import torchvision
import imageio

from concurrent.futures import ThreadPoolExecutor
from math import tanh
from time import sleep, time
from tqdm import tqdm
from torch.optim import Adam
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from rtpt import RTPT

from xrl.agents.policies.dreamerv2.dataset import ModelDataset
from xrl.agents.policies.dreamerv2.model import WorldModel, Actor, Critic, LossModel, ActorLoss, CriticLoss

import xrl.utils.utils as xutils
# plt.ion()

# world model parameter
batch = 50
L = 50 #seq len world training
history_size = 400
lr_world = 2e-4

# behavior parameter
H = 15 #imagination length
gamma = 0.995 #discount factor
lamb = 0.95 #lambda-target
lr_actor = 4e-5
lr_critic = 1e-4
target_interval = 100 #update interval for target critic

# common parameter
gradient_clipping = 100
adam_eps = 1e-5
decay = 1e-6


def act_straight_through(z_hat_sample, actor):
    a_logits = actor(z_hat_sample)
    a_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
        logits=a_logits
    ).sample()
    a_probs = torch.softmax(a_logits, dim=-1)
    a_sample = a_sample + a_probs - a_probs.detach()

    return a_sample, a_logits


# training function
def train(cfg):
    ### HYPERPARAMETERS ###
    print('Experiment name:', cfg.exp_name)
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
    env = gym.make(cfg.env_name, frameskip = 4)
    num_actions = env.action_space.n

    ### MODELS ###
    world = WorldModel(gamma, num_actions).to(device)
    actor = Actor(num_actions).to(device)
    critic = Critic().to(device)
    target = Critic().to(device)

    criterionModel = LossModel()
    criterionActor = ActorLoss()
    criterionCritic = CriticLoss()

    optim_model = Adam(world.parameters(), lr=lr_world, eps=adam_eps, weight_decay=decay)
    optim_actor = Adam(actor.parameters(), lr=lr_actor, eps=adam_eps, weight_decay=decay)
    optim_critic = Adam(critic.parameters(), lr=lr_critic, eps=adam_eps, weight_decay=decay)
    optim_target = Adam(target.parameters())

    i_episode = 1
    steps_done = [0]

    if os.path.isfile(save_path):
        print("Trying to load", save_path)
        w = torch.load(save_path, map_location=torch.device(device))
        try:
            world.load_state_dict(w["world"])
            optim_model.load_state_dict(w["optim_model"])
            actor.load_state_dict(w["actor"])
            optim_actor.load_state_dict(w["optim_actor"])
            critic.load_state_dict(w["critic"])
            optim_critic.load_state_dict(w["optim_critic"])
            criterionActor = ActorLoss(*w["criterionActor"])
            i_episode = w['episode']
            steps_done = w['steps_done']
        except:
            print("error loading model")
            world = WorldModel(gamma, num_actions).to(device)
            actor = Actor(num_actions).to(device)
            critic = Critic().to(device)
            target = Critic().to(device)
            criterionModel = LossModel()
            criterionActor = ActorLoss()
            criterionCritic = CriticLoss()
            optim_model = Adam(world.parameters(), lr=lr_world, eps=adam_eps, weight_decay=decay)
            optim_actor = Adam(actor.parameters(), lr=lr_actor, eps=adam_eps, weight_decay=decay)
            optim_critic = Adam(critic.parameters(), lr=lr_critic, eps=adam_eps, weight_decay=decay)
            optim_target = Adam(target.parameters())
            i_episode = 1
            steps_done = [0]
        del w
    else:
        print(save_path, "does not exists, starting fresh")
    with torch.no_grad():
        target.load_state_dict(critic.state_dict())

    ### MISC ###
    resize = torchvision.transforms.Resize(
        (64,64),
        interpolation=PIL.Image.BICUBIC
    )
    grayscale = torchvision.transforms.Grayscale()
    def transform_obs(obs):
        obs = resize(
            torch.from_numpy(obs.transpose(2,0,1))
        ) #3, 64, 64
        return (obs.float() - 255/2).unsqueeze(0)
    history = []
    tensor_range = torch.arange(0, num_actions).unsqueeze(0)
    random_action_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
        torch.ones((1,num_actions))
    )

    # inner function to give it access to shared variables
    def gather_episode():
        with torch.no_grad():
            rtpt_s = RTPT(name_initials='DV', experiment_name=cfg.exp_name + "_el",
                    max_iterations=cfg.train.num_episodes)
            rtpt_s.start()
            while True:
                obs = env.reset()
                obs = transform_obs(obs)
                episode = [obs]
                obs = obs.to(device)
                z_sample, h = world(None, obs, None, inference=True)
                done = False
                r_sum = 0
                t = 0
                while t < 50000:
                    a = actor(z_sample)
                    a = torch.distributions.one_hot_categorical.OneHotCategorical(
                        logits = a
                    ).sample()
                    obs, rew, done, _ = env.step(int((a.cpu()*tensor_range).sum().round()))
                    obs = transform_obs(obs)
                    obs = obs.to(device)
                    episode.extend([a.cpu(), tanh(rew), done, obs.cpu()])
                    r_sum += rew
                    if not done:
                        z_sample, h = world(a, obs, z_sample.reshape(-1, 1024), h, inference=True)
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
    while len(history) < 50:
        # check every second if first history entry is inside
        print("Wainting until history large enough (current size: {}, needed: {})".format(len(history), 50), end="\r")
        sleep(5.0)
    print("done")
    ### DATASET ###
    ds = ModelDataset(history, seq_len=L, gamma=gamma, history_size=history_size)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    start = time()
    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                    max_iterations=cfg.train.num_episodes)
    rtpt.start()
    while steps_done[0] < cfg.train.max_steps:
        l_world = 0
        l_actor = 0
        l_critic = 0
        len_h = 0
        last_rew = 0
        pbar = tqdm(loader)
        for s, a, r, g in pbar:
            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            g = g.to(device)
            z_list = []
            h_list = []

            ### Train world model ###
            z_logit, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, _ = world(
                a=None,
                x=s[:,0],
                z=None,
                h=None
            )
            loss_model = criterionModel(
                s[:,0],
                r[:,0], #false r_0 does not exist, 0: t=1 but expect 0:t=0
                g[:,0], #same but ok since never end of episode
                z_logit,
                z_sample,
                x_hat,
                0, #rhat
                gamma_hat,
                z_hat_logits
            )
            z_list.append(z_sample.detach())
            h_list.append(h.detach())
            for t in range(L):
                z_logit, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, _ = world(
                    a[:,t],
                    s[:,t+1],
                    z_sample,
                    h
                )
                z_list.append(z_sample.detach())
                h_list.append(h.detach())
                loss_model += criterionModel(
                    s[:,t+1],
                    r[:,t], #r time array starts at 1; 0: t=1
                    g[:,t], #g time array starts at 1; 0: t=1
                    z_logit,
                    z_sample,
                    x_hat,
                    r_hat,
                    gamma_hat,
                    z_hat_logits
                )

            loss_model /= L
            loss_model.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(), gradient_clipping)
            optim_model.step()
            optim_model.zero_grad()

            ### Train actor critic ###
            #store every value to compute V since we sum backwards
            r_hat_sample_list = []
            gamma_hat_sample_list = []
            a_sample_list = []
            a_logits_list = []

            z_hat_sample = torch.cat(z_list, dim=0).detach() #convert all z to z0, squash time dim
            z_hat_sample_list = [z_hat_sample]

            h = torch.cat(h_list, dim=0).detach() #get corresponding h0

            # store values
            for _ in range(H):
                a_sample, a_logits = act_straight_through(z_hat_sample, actor)

                *_, h, (z_hat_sample, r_hat_sample, gamma_hat_sample) = world(
                    a_sample,
                    x = None,
                    z = z_hat_sample.reshape(-1, 1024),
                    h = h,
                    dream=True
                )
                r_hat_sample_list.append(r_hat_sample)
                gamma_hat_sample_list.append(gamma_hat_sample)
                z_hat_sample_list.append(z_hat_sample)
                a_sample_list.append(a_sample)
                a_logits_list.append(a_logits)

            # calculate paper recursion by looping backward 
            V = r_hat_sample_list[-1] + gamma_hat_sample_list[-1] * target(z_hat_sample_list[-1]) #V_H-1
            ve = critic(z_hat_sample_list[-2].detach())
            loss_critic = criterionCritic(V.detach(), ve)
            loss_actor = criterionActor(
                a_sample_list[-1],
                torch.distributions.one_hot_categorical.OneHotCategorical(
                    logits=a_logits_list[-1], validate_args=False
                ),
                V,
                ve.detach()
            )
            for t in range(H-2, -1, -1):
                V = r_hat_sample_list[t] + gamma_hat_sample_list[t] * ((1-lamb)*target(z_hat_sample_list[t+1]) + lamb*V)
                ve = critic(z_hat_sample_list[t].detach())
                loss_critic += criterionCritic(V.detach(), ve)
                loss_actor += criterionActor(
                    a_sample_list[t],
                    torch.distributions.one_hot_categorical.OneHotCategorical(
                        logits=a_logits_list[t], validate_args=False
                    ),
                    V,
                    ve.detach()
                )

            loss_actor /= (H-1)
            loss_critic /= (H-1)

            #update actor
            loss_actor.backward()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clipping)
            optim_actor.step()
            optim_actor.zero_grad()
            optim_model.zero_grad()

            #update critic
            torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clipping)
            optim_critic.step()
            optim_critic.zero_grad()
            optim_target.zero_grad()

            #update target network with critic weights
            i_episode += 1
            if not i_episode % target_interval:
                with torch.no_grad():
                    target.load_state_dict(critic.state_dict())

            # set logging variables
            l_world = loss_model.item()
            l_actor = loss_actor.item()
            l_critic = loss_critic.item()
            len_h = len(history)
            last_rew = sum(history[-1][2::4]) * (1 / (np.tanh(1)))
            # display
            pbar.set_postfix(
                l_world = l_world,
                l_actor = l_actor,
                l_critic = l_critic,
                len_h = len_h,
                i_episode = i_episode,
                last_rew = last_rew,
            )
            #print(a_logits_list[0][0].detach())
            #print(list(z_hat_sample_list[-1][0,1].detach().cpu().numpy().round()).index(1))

        # log all
        writer.add_scalar('Train/Loss World Model', l_world, i_episode)
        writer.add_scalar('Train/Loss Actor', l_actor, i_episode)
        writer.add_scalar('Train/Loss Critic', l_critic, i_episode)

        #save once in a while
        if i_episode % cfg.train.save_every == 0:
            print("Saving...")
            torch.save(
                {
                    "world":world.state_dict(),
                    "actor":actor.state_dict(),
                    "critic":critic.state_dict(),
                    "optim_model":optim_model.state_dict(),
                    "optim_actor":optim_actor.state_dict(),
                    "optim_critic":optim_critic.state_dict(),
                    "criterionActor": (criterionActor.ns, criterionActor.nd, criterionActor.ne),
                    "episode":i_episode,
                    "steps_done":steps_done
                },
                save_path
            )
            print("...done")
            img = np.clip((x_hat[0].detach().cpu().numpy().transpose(1,2,0))/255+0.5, 0, 1)
            #image_path = "xrl/dreamerv2/world.png"
            #plt.imsave(image_path, img)
            writer.add_image('x_hat', img.transpose(2, 0, 1), i_episode)

        # plt.figure(1)
        # # plt.clf()
        # plt.imshow(x_hat[0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
        # # plt.pause(0.001)
        # plt.show()
        i_episode += 1
        rtpt.step()


# eval function
def eval(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Eval mode')
    print('Env name:', cfg.env_name)
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
            print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)
    
    # params for saving loading
    save_path = os.getcwd() + "/xrl/checkpoints/"
    save_path = save_path + cfg.exp_name + "_model.pth"
    # device to use for training
    device = cfg.device
    # params for game
    env = gym.make(cfg.env_name, frameskip = 4)
    num_actions = env.action_space.n
    ### MISC ###
    resize = torchvision.transforms.Resize(
        (64,64),
        interpolation=PIL.Image.BICUBIC
    )
    def transform_obs(obs):
        obs = resize(
            torch.from_numpy(obs.transpose(2,0,1))
        ) #3, 64, 64
        return (obs.float() - 255/2).unsqueeze(0)
    tensor_range = torch.arange(0, num_actions).unsqueeze(0)
    # models
    world = WorldModel(gamma, num_actions).to(device)
    actor = Actor(num_actions).to(device)
    if os.path.isfile(save_path):
        print("Trying to load", save_path)
        w = torch.load(save_path, map_location=torch.device(device))
        world.load_state_dict(w["world"])
        actor.load_state_dict(w["actor"])
    # game loop
    obs = env.reset()
    obs = transform_obs(obs)
    obs = obs.to(device)
    z_sample, h = world(None, obs, None, inference=True)
    done = False
    rsum = 0
    with torch.no_grad():
        for t in count():
            a = actor(z_sample)
            a = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits = a
            ).sample()
            obs, rew, done, _ = env.step(int((a.cpu()*tensor_range).sum().round()))
            #xutils.plot_screen(env, "eval", t)
            rsum += rew
            obs = transform_obs(obs)
            obs = obs.to(device)
            if not done:
                z_sample, h = world(a, obs, z_sample.reshape(-1, 1024), h, inference=True)
            print("Step: {}, Reward: {}".format(t, rsum), end="\r")
            if done:
                break
        print("Final - Step: {}, Reward: {}".format(t, rsum))
