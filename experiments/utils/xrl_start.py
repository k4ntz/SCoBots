import torch
import matplotlib.pyplot as plt
import numpy as np

from torchinfo import summary
from tqdm import tqdm
from rtpt import RTPT

from algos import reinforce
from algos import genetic_rl as genetic
from scobi import Environment
from captum.attr import IntegratedGradients
from pathlib import Path
import matplotlib as mpl
import os
import json
try:
    from experiments.utils.xrl_explain import explain_agent
except:
    pass
from experiments.utils.xrl_play_agent import play_agent
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from experiments.utils.xrl_utils import Drawer
from experiments.my_normalizer import load_normalizer
from ppo_utils import ppo_load_model_v2, ppo_select_action, load_ppo_env

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#def reinforce_select_action(features, policy, random_tr = -1, n_actions=3):
#    policy.eval()
#    with torch.no_grad():
#        features = torch.tensor(features, device=dev, dtype=torch.float32).unsqueeze(0)
#        probs = policy(features)
#        action = torch.argmax(probs)
#        return action.item(), None, None

# function to call reinforce algorithm
def use_reinforce(cfg, mode):
    if mode == "train":
        if "Kangaroo" in cfg.env_name:
            reinforce.train_kangaroo(cfg)
        else:
            reinforce.train(cfg)
    elif mode == "eval":
        model, normalizer, epochs = reinforce.eval_load(cfg)
        model.eval()
        play_agent(cfg, model, reinforce.select_action, normalizer, epochs)
    elif mode == "discover":
        reinforce.eval_reward_discovery(cfg)
    elif mode == "explain":
        _, normalizer, _ = reinforce.eval_load(cfg)
        rule_env = "freeway"
        ruleset_path = f"remix_data/{rule_env}/output.rules"
        explain_agent(cfg, normalizer, ruleset_path)

def use_ppo(cfg, mode):
    if mode == "train":
        print(f"Mode {mode} not implemented for PPO")
    elif mode == "eval":
        base_path = "../ppo_pruned_inx64xout"
        normalizer = load_normalizer(os.path.join(base_path, "normalizer.json"))
        model = ppo_load_model_v2(cfg)
        env = load_ppo_env()
        play_agent(cfg, model, ppo_select_action, normalizer, None, env=env)
    elif mode == "discover":
        print(f"Mode {mode} not implemented for PPO")
    elif mode == "explain":
        base_path = "../ppo_pruned_inx64xout"
        normalizer = load_normalizer(os.path.join(base_path, "normalizer.json"))
        env = load_ppo_env()
        ruleset_path = f"{base_path}/output.rules"
        explain_agent(cfg, normalizer, ruleset_path, env=env)



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
    elif cfg.rl_algo == 3:
        print("Selected algorithm: PPO")
        use_ppo(cfg, mode)
    else:
        print("Unknown algorithm selected")