# main file for all rl algos

import random
import torch
import numpy as np

from captum.attr import IntegratedGradients
from torch.distributions import Categorical
from torchinfo import summary

from xrl.algorithms import reinforce
from xrl.algorithms import genetic_rl as genetic
from xrl.algorithms import dreamer_v2
from xrl.algorithms import minidreamer

from xrl.utils.utils import get_config
# otherwise genetic loading model doesnt work, torch bug?
from xrl.genetic_rl import policy_net
import xrl.utils.plotter as plt
import xrl.utils.utils as xutils
import xrl.utils.video_logger as vlogger
from xrl.environments import agym

# helper function to select action from loaded agent
# has random probability parameter to test stability of agents
def select_action(features, policy, random_tr = -1, select_argmax=False):
    sample = random.random()
    if sample > random_tr:
        # calculate probabilities of taking each action
        probs = policy(features)
        if select_argmax:
            return probs.argmax().item()
        # sample an action from that set of probs
        else:
            sampler = Categorical(probs)
            action = sampler.sample()
    else:
        action = random.randint(0, 5)
    # return action
    return action


# function to test agent loaded via main switch
def play_agent(agent, cfg):
    # init env
    env = agym.make(cfg.env_name)
    _, ep_reward = env.reset(), 0
    _, _, done, _ = env.step(1)
    raw_features, features, _, _ = xutils.do_step(env)
    # only when raw features should be used
    if cfg.train.use_raw_features:
        features = np.array(np.array([[0,0] if x==None else x for x in raw_features]).tolist()).flatten()
    # init objects
    summary(agent, input_size=(1, len(features)), device=cfg.device)
    logger = vlogger.VideoLogger(size=(480, 480))
    ig = IntegratedGradients(agent)
    ig_sum = []
    ig_action_sum = []
    l_features = []
    feature_titles = plt.get_feature_titles(int(len(raw_features)/2))
    # env loop
    plotter = plt.Plotter()
    t = 0
    while t < 3000:  # Don't infinite loop while playing
        # only when raw features should be used
        if cfg.train.use_raw_features:
            features = np.array(np.array([[0,0] if x==None else x for x in raw_features]).tolist()).flatten()
        features = torch.tensor(features).unsqueeze(0).float().to(cfg.device)
        action = select_action(features, agent)
        if cfg.liveplot or cfg.make_video:
            img = plotter.plot_IG_img(ig, cfg.exp_name, features, feature_titles, action, env, cfg.liveplot)
            logger.fill_video_buffer(img)
        else:
            ig_sum.append(plt.get_integrated_gradients(ig, features, action))
            ig_action_sum.append(np.append(plt.get_integrated_gradients(ig, features, action), [action]))
        print('Reward: {:.2f}\t Step: {:.2f}'.format(
                ep_reward, t), end="\r")
        raw_features, features, reward, done = xutils.do_step(env, action, raw_features)
        l_features.append(features)
        ep_reward += reward
        t += 1
        if done:
            print("\n")
            break
    if cfg.liveplot or cfg.make_video:
        logger.save_video(cfg.exp_name)
        print('Final reward: {:.2f}\tSteps: {}'.format(
        ep_reward, t))
    else:
        ig_sum = np.asarray(ig_sum)
        ig_action_sum = np.asarray(ig_action_sum)
        print('Final reward: {:.2f}\tSteps: {}\tIG-Mean: {}'.format(
        ep_reward, t, np.mean(ig_sum, axis=0)))

    ################## PLOT STUFF ##################
    #xutils.ig_pca(ig_action_sum, env.unwrapped.get_action_meanings())
    #xutils.plot_igs_violin(ig_action_sum, feature_titles, env.unwrapped.get_action_meanings())
    #if not cfg.train.make_hidden:
    #    # plot some weight stuff due of linear model
    #    xutils.plot_lin_weights(agent, feature_titles, env.unwrapped.get_action_meanings())
    #xutils.plot_igs(ig_action_sum, feature_titles, env.unwrapped.get_action_meanings())


# function to call reinforce algorithm
def use_reinforce(cfg):
    print("Selected algorithm: REINFORCE")
    if cfg.mode == "train":
        reinforce.train(cfg)
    elif cfg.mode == "eval":
        agent = reinforce.eval_load(cfg)
        play_agent(agent=agent, cfg=cfg)


# function to call deep neuroevolution algorithm
def use_genetic(cfg):
    print("Selected algorithm: Deep Neuroevolution")
    if cfg.mode == "train":
        genetic.train(cfg)
    elif cfg.mode == "eval":
        agent = genetic.eval_load(cfg)
        play_agent(agent=agent, cfg=cfg)


# function to call dreamerv2
def use_dreamerv2(cfg):
    print("Selected algorithm: DreamerV2")
    if cfg.mode == "train":
        dreamer_v2.train(cfg)
    elif cfg.mode == "eval":
        dreamer_v2.eval(cfg)


# function to call minidreamer
def use_minidreamer(cfg):
    print("Selected algorithm: Minidreamer")
    if cfg.mode == "train":
        minidreamer.train(cfg)
    elif cfg.mode == "eval":
        print("Not implemented ...")


# main function
# switch for each algo
if __name__ == '__main__':
    cfg = get_config()
    # algo selection
    # 1: REINFORCE
    # 2: Deep Neuroevolution
    # 3: DreamerV2
    # 4: Minidreamer
    if cfg.rl_algo == 1:
        use_reinforce(cfg)
    elif cfg.rl_algo == 2:
        use_genetic(cfg)
    elif cfg.rl_algo == 3:
        use_dreamerv2(cfg)
    elif cfg.rl_algo == 4:
        use_minidreamer(cfg)
    else:
        print("Unknown algorithm selected")
