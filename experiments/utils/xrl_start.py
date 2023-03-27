import torch
import matplotlib.pyplot as plt
import numpy as np

from torchinfo import summary
from tqdm import tqdm
from rtpt import RTPT
from time import perf_counter

from algos import reinforce
from algos import genetic_rl as genetic
from scobi import Environment
from captum.attr import IntegratedGradients
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None' 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def re_select_action(*args, **kwargs):
    return reinforce.select_action(*args, **kwargs)





# function to test agent loaded via main switch
def play_agent(cfg, model, select_action_func, normalizer):
    # init env

    env = Environment(cfg.env_name,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file,
                      draw_features=True)
    n_actions = env.action_space.n
    _, ep_reward = env.reset(), 0
    obs, _, _, _, _, info, obs_raw = env.step(1)
    features = obs
    summary(model, input_size=(1, len(features)), device=cfg.device)
    runs = 3
    print("Runs:", runs)
    rewards = []
    all_sco_rewards = []
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name + "_EVAL", max_iterations=runs)
    rtpt.start()
    model.to(dev)
    ig = IntegratedGradients(model)
    fps = 30
    frame_delta = 1.0 / fps
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fsize = (obs_raw.shape[1]+0) * px, (obs_raw.shape[0]+ 0) * px
    fig = plt.figure(figsize=fsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    img = ax.imshow(obs_raw, interpolation='none')
    plt.ion()
    for run in tqdm(range(runs)):
        # env loop
        t = 0
        ep_reward = 0
        sco_reward = 0
        env.reset()
        while t < cfg.train.max_steps_per_trajectory:  # Don't infinite loop while playing
            features = normalizer.normalize(features)
            action, _, probs = select_action_func(features, model, -1, n_actions)
            input = torch.tensor(features, requires_grad=True).unsqueeze(0).to(dev)
            output = int(np.argmax(probs[0]))
            attris = ig.attribute(input, target=output, method="gausslegendre")
            if cfg.liveplot:
                img.set_data(obs_raw)
                plt.pause(frame_delta)  
            env.set_feature_attribution(attris.squeeze(0).detach().cpu().numpy())
            obs, reward, scobi_reward, done, done2, info, obs_raw = env.step(action)
            features = obs
            ep_reward += reward
            sco_reward += scobi_reward
            t += 1
            if done or done2:
                break
        rewards.append(ep_reward)
        all_sco_rewards.append(sco_reward)
        rtpt.step()
    print(rewards)
    print(all_sco_rewards)
    print("Mean of Env Rewards:", sum(rewards) / len(rewards))




# function to call reinforce algorithm
def use_reinforce(cfg, mode):
    if mode == "train":
        reinforce.train(cfg)
    else:
        if mode == "eval":
            model, normalizer = reinforce.eval_load(cfg)
            play_agent(cfg, model, re_select_action, normalizer)
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