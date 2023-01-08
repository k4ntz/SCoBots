import numpy as np
import os
import random
import random
import time
import torch
import torch.optim as optim
import utils.utils as xutils
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm
from rtpt import RTPT
from xrl.environments import env_manager
from xrl.agents import Agent
from networks import FC_Normed_Net

EPS = np.finfo(np.float32).eps.item()
PATH_TO_OUTPUTS = os.getcwd() + "/xrl/checkpoints/"
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
    ret = 0
    entropy_alpha = cfg.train.entropy_alpha #0 if not set in cfg
    policy_loss = []
    returns = []
    for r, e in zip(rewards[::-1], entropies[::-1]):
        # discounted return including entropy regularization
        ret = r + entropy_alpha * e + cfg.train.gamma * ret
        returns.insert(0, ret)

    returns = torch.tensor(returns, device=dev)
    returns = (returns - returns.mean()) / (returns.std() + EPS)
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append((-log_prob * R))
    optimizer.zero_grad()

    policy_loss = torch.cat(policy_loss).sum() #was mean
    policy_loss.backward()
    optimizer.step()
    episode_entropy = np.mean(entropies)
    return policy, optimizer, policy_loss, episode_entropy


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
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
    print('Experiment name:', cfg.exp_name)
    torch.manual_seed(cfg.seed)
    print('Seed:', torch.initial_seed())
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    # init env to get params for policy net
    env = env_manager.make(cfg, True)
    n_actions = env.action_space.n
    gametype = xutils.get_gametype(env)
    env.reset()
    obs, _, _, _, info = env.step(1)
    raw_features = agent.image_to_feature(obs, info, gametype)
    features = agent.feature_to_mf(raw_features)
    print('Action space: ' + str(env._env.unwrapped.get_action_meanings()))
    print("Feature Vector Length:", len(features))
    print("Hidden Layer:", cfg.train.make_hidden)

    # init fresh policy and optimizer
    policy = FC_Normed_Net(len(features), cfg.train.hidden_layer_size, n_actions, cfg.train.make_hidden).to(dev)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.train.learning_rate)
    i_episode = 1
    # overwrite if checkpoint exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        policy, optimizer, i_episode = load_model(model_path, policy, optimizer)
        i_episode += 1

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
    nr_buffer = 0
    pnl_buffer = 0
    pne_buffer = 0
    step_buffer = 0

    # training loop
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name, max_iterations=cfg.train.num_episodes)
    rtpt.start()
    while i_episode < cfg.train.num_episodes:
        env.reset()
        rewards = []
        entropies = []
        log_probs = []
        ep_natural_reward = 0
        int_duration = 0
        t = 0
        start_time = time.perf_counter()
        while t < cfg.train.max_steps:
            # interaction
            int_s_time = time.perf_counter()
            action, log_prob, probs = agent.mf_to_action(features, agent.model, cfg.train.random_action_p, n_actions)
            obs, natural_reward, terminated, truncated, info = env.step(action)
            raw_features = agent.image_to_feature(obs, info, gametype)
            features = agent.feature_to_mf(raw_features)
            int_duration += time.perf_counter() - int_s_time

            # collection
            entropy = -np.sum(list(map(lambda p : p * (np.log(p) / np.log(n_actions)) if p[0] != 0 else 0, probs)))
            log_probs.append(log_prob)
            rewards.append(natural_reward)
            entropies.append(entropy)
            ep_natural_reward += natural_reward
            t += 1
            if terminated or truncated:
                break
        # policy update
        bp_s_time = time.perf_counter()
        policy, optimizer, loss, ep_entropy = finish_episode(policy, optimizer, cfg, log_probs, rewards, entropies)
        bp_duration = time.perf_counter() - bp_s_time

        # checkpointing
        if (i_episode) % cfg.train.save_every == 0:
            save_policy(cfg.exp_name, policy, i_episode, optimizer)

        # update logging data
        ep_duration = time.perf_counter() - start_time
        if running_reward is None:
            running_reward = ep_natural_reward
        else:
            running_reward = 0.05 * ep_natural_reward + (1 - 0.05) * running_reward
        nr_buffer += ep_natural_reward
        pnl_buffer += loss.detach()
        pne_buffer += ep_entropy
        step_buffer += t

        # episode stats
        print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tEntropy: {:.2f}\tSteps: {}\tDuration: {:.2f} [ENV: {:.2f} | BP: {:.2f}]'.format(
            i_episode, ep_natural_reward, running_reward, ep_entropy, t, ep_duration, int_duration, bp_duration))
        # tfboard logging
        if i_episode % cfg.train.log_steps == 0:
            avg_nr = nr_buffer / cfg.train.log_steps
            avg_pnl = pnl_buffer / cfg.train.log_steps
            avg_pne = pne_buffer / cfg.train.log_steps
            avg_step = step_buffer / cfg.train.log_steps
            writer.add_scalar('rewards/avg natural', avg_nr, i_episode)
            writer.add_scalar('loss/avg_policy_net', avg_pnl, i_episode)
            writer.add_scalar('loss/avg_policy_net_entropy', avg_pne, i_episode)
            writer.add_scalar('various/avg_steps', avg_step, i_episode)
            nr_buffer = 0
            pnl_buffer = 0
            pne_buffer = 0
            step_buffer = 0
        i_episode += 1
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
    obs, _, _, _, info = env.step(1)
    raw_features = agent.image_to_feature(obs, info, xutils.get_gametype(env))
    features = agent.feature_to_mf(raw_features)
    print("Make hidden layer in nn:", cfg.train.make_hidden)
    policy = FC_Normed_Net(len(features), cfg.train.hidden_layer_size, n_actions, cfg.train.make_hidden)
    # load if exists
    model_path = model_name(cfg.exp_name + "-seed" + str(cfg.seed))
    if os.path.isfile(model_path):
        policy, _, i_episode = load_model(model_path, policy)
        print('Episodes trained:', i_episode)
    policy.eval()
    # return policy 
    return policy
