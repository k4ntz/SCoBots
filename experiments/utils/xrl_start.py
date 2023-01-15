import torch
import matplotlib.pyplot as plt

from torchinfo import summary
from tqdm import tqdm
from rtpt import RTPT

from algos import reinforce
from algos import genetic_rl as genetic
from scobi import Environment

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        while t < cfg.train.max_steps_per_trajectory:  # Don't infinite loop while playing
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
        #TODO: silent mode not working for genetic only
        genetic.train(cfg)
    else:
        model, gen_select_action = genetic.eval_load(cfg)
        if mode == "eval":
            play_agent(cfg, model, gen_select_action)
        elif mode == "explain":
            pass
            # explain(agent=agent, cfg=cfg)


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