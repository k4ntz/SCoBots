import numpy as np
import os
import random
import random
import time
import torch
import torch.optim as optim
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm
from rtpt import RTPT
from scobi import Environment
from . import networks

#from agent import Agent

EPS = np.finfo(np.float32).eps.item()
PATH_TO_OUTPUTS = os.getcwd() + "/checkpoints/"
if not os.path.exists(PATH_TO_OUTPUTS):
    os.makedirs(PATH_TO_OUTPUTS)

model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_model.pth"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select_action(features, policy, random_tr = -1, n_actions=3):
    input = torch.tensor(features).unsqueeze(0).float().to(dev)
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
    return action, log_prob, probs.detach().cpu().numpy()


def finish_episode(policy, optimizer, cfg, log_probs, rewards, entropies):
    R = 0
    entropy_alpha = cfg.train.entropy_alpha #0 if not set in cfg
    policy_loss = []
    returns = []
    for r, e in zip(rewards[::-1], entropies[::-1]):
        # discounted reward including entropy regularization
        R = r + entropy_alpha * e + cfg.train.gamma * R 
        returns.insert(0, R)

    returns = torch.tensor(returns, device=dev)
    returns = (returns - returns.mean()) / (returns.std() + EPS)
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append((-log_prob * R))
    optimizer.zero_grad()

    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.clip_norm)
    optimizer.step()
    episode_entropy = np.mean(entropies)
    return policy, optimizer, policy_loss, episode_entropy


# save model helper function
def save_policy(training_name, policy, episode, optimizer):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    #print("Saving {}".format(model_path))
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


def train(cfg):
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
    torch.manual_seed(cfg.seed)
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    # init env to get params for policy net
    env = Environment(cfg.env_name, interactive=cfg.scobi_interactive, focus_dir=cfg.scobi_focus_dir, focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, info, _ = env.step(1)
    print("EXPERIMENT")
    print(">> Selected algorithm: REINFORCE")
    print('>> Experiment name:', cfg.exp_name)
    print('>> Seed:', torch.initial_seed())
    print(">> Random Action probability:", cfg.train.random_action_p)
    print('>> Gamma:', cfg.train.gamma)
    print('>> Learning rate:', cfg.train.learning_rate)
    print("ENVIRONMENT")
    print('>> Action space: ' + str(env.action_space_description))
    print(">> Observation Vector Length:", len(obs))

    # init fresh policy and optimizer
    policy = networks.FC_Net(len(obs), cfg.train.hidden_layer_size, n_actions).to(dev)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.train.learning_rate)
    i_epoch = 1
    # overwrite if checkpoint exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        policy, optimizer, i_epoch = load_model(model_path, policy, optimizer)
        i_epoch += 1

    print("TRAINING")
    print('>> Epochs:', cfg.train.num_episodes)
    print('>> Steps per Epoch:', cfg.train.steps_per_episode)
    print('>> Logging Interval (Steps):', cfg.train.log_steps)
    print('>> Checkpoint Interval (Epochs):', cfg.train.save_every)
    print('>> Current Epoch:', i_epoch)
    print("Training started...")
    # reinit agent with loaded policy model
    running_return = None
    # tfboard logging buffer
    tfb_nr_buffer = 0
    tfb_pnl_buffer = 0
    tfb_pne_buffer = 0
    tfb_step_buffer = 0
    tfb_policy_updates_counter = 0


    # training loop
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name, max_iterations=cfg.train.num_episodes)
    rtpt.start()
    while i_epoch <= cfg.train.num_episodes:
        stdout_nr_buffer = 0
        stdout_pnl_buffer = 0
        stdout_pne_buffer = 0
        stdout_step_buffer = 0
        stdout_policy_updates_counter = 0
        sum_ep_duration = 0
        sum_int_duration = 0
        sum_pol_duration = 0
        i_episode_step = 0
        while i_episode_step < cfg.train.steps_per_episode:
            env.reset()
            rewards = []
            entropies = []
            log_probs = []
            ep_return = 0
            int_duration = 0
            i_trajectory_step = 0
            incomplete_traj = False
            int_s_time = time.perf_counter()
            while i_trajectory_step < cfg.train.max_steps_per_trajectory:
                # interaction
                action, log_prob, probs = select_action(obs, policy, cfg.train.random_action_p, n_actions)
                obs, natural_reward, terminated, truncated, info, _ = env.step(action)

                # collection
                entropy = -np.sum(list(map(lambda p : p * (np.log(p) / np.log(n_actions)) if p[0] != 0 else 0, probs)))
                log_probs.append(log_prob)
                rewards.append(natural_reward)
                entropies.append(entropy)
                ep_return += natural_reward
                i_trajectory_step += 1
                i_episode_step += 1

                # tfboard logging
                if i_episode_step % cfg.train.log_steps == 0 and tfb_policy_updates_counter > 0:
                    global_step = (i_epoch - 1) * cfg.train.steps_per_episode + i_episode_step
                    avg_nr = tfb_nr_buffer / tfb_policy_updates_counter
                    avg_pnl = tfb_pnl_buffer / tfb_policy_updates_counter
                    avg_pne = tfb_pne_buffer / tfb_policy_updates_counter
                    avg_step = tfb_step_buffer / tfb_policy_updates_counter
                    writer.add_scalar('rewards/avg_return', avg_nr, global_step)
                    writer.add_scalar('loss/avg_policy_net', avg_pnl, global_step)
                    writer.add_scalar('loss/avg_policy_net_entropy', avg_pne, global_step)
                    writer.add_scalar('various/avg_steps', avg_step, global_step)
                    tfb_nr_buffer = 0
                    tfb_pnl_buffer = 0
                    tfb_pne_buffer = 0
                    tfb_step_buffer = 0
                    tfb_policy_updates_counter = 0

                # break conditions
                if terminated or truncated:
                    break
                if i_episode_step == cfg.train.steps_per_episode:
                    incomplete_traj = True
                    break
            
            # policy update
            int_duration += time.perf_counter() - int_s_time
            pol_s_time = time.perf_counter()
            policy, optimizer, loss, ep_entropy = finish_episode(policy, optimizer, cfg, log_probs, rewards, entropies)
            pol_duration = time.perf_counter() - pol_s_time
            ep_duration = int_duration + pol_duration

            if not incomplete_traj:
                tfb_policy_updates_counter += 1
                tfb_nr_buffer += ep_return
                tfb_pnl_buffer += loss.detach()
                tfb_pne_buffer += ep_entropy
                tfb_step_buffer += i_trajectory_step

                stdout_policy_updates_counter += 1
                stdout_nr_buffer += ep_return
                stdout_pnl_buffer += loss.detach()
                stdout_pne_buffer += ep_entropy
                stdout_step_buffer += i_trajectory_step

                sum_ep_duration += ep_duration
                sum_int_duration += int_duration
                sum_pol_duration += pol_duration
                # update logging data
                if running_return is None:
                    running_return = ep_return
                else:
                    running_return = 0.05 * ep_return + (1 - 0.05) * running_return



        # checkpointing
        checkpoint_str = ""
        if i_epoch % cfg.train.save_every == 0:
            save_policy(cfg.exp_name, policy, i_epoch, optimizer)
            checkpoint_str = "checkpoint"

        # episode stats
        c = stdout_policy_updates_counter
        print('Epoch {}:\tRunning Return: {:.2f}\tavgReturn: {:.2f}\tavgEntropy: {:.2f}\tavgObjFuncValue: {:.2f}\tavgSteps: {:.2f}\tDuration: {:.2f} [ENV: {:.2f} | P_UPDATE: {:.2f}]\t{}'.format(
            i_epoch, running_return, stdout_nr_buffer / c, stdout_pne_buffer / c, loss / c, stdout_step_buffer / c, sum_ep_duration, sum_int_duration, sum_pol_duration, checkpoint_str))
        
        i_epoch += 1
        rtpt.step()


# eval function, returns trained model
def eval_load(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Evaluating Mode')
    print('Seed:', cfg.seed)
    print("Random Action probability:", cfg.train.random_action_p)
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)
    # init env
    env = Environment(cfg.env_name, focus_dir="focusfiles")
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, info, _ = env.step(1)
    print("Make hidden layer in nn:", cfg.train.make_hidden)
    policy = networks.FC_Net(len(obs), cfg.train.hidden_layer_size, n_actions).to(dev)
    # load if exists
    model_path = model_name(cfg.exp_name + "-seed" + str(cfg.seed))
    if os.path.isfile(model_path):
        policy, _, i_episode = load_model(model_path, policy)
        print('Episodes trained:', i_episode)
    policy.eval()
    return policy
