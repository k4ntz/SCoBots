import subprocess
import os
import numpy as np
import pandas as pd
from collections import defaultdict


MAX_ROWS = 5
# load a .npz file and return the data
def load_npz(file):
    data = np.load(file)
    # print the keys
    return data

def load_csv(file):
    data = pd.read_csv(file)
    return data

games = ["Pong", "Boxing"]
for game in games:

    result_dict = defaultdict(lambda: defaultdict(lambda: np.nan))
    for file in os.listdir("eclaire_configs"):
        if not file.endswith(".yaml"):
            continue
        _, rest = file.split("_",1)
        exp_folder, yaml_str = rest.split(".",1)
        _ , exp_name = exp_folder.split("_",1)

        if game not in exp_name:
            continue

        res = load_csv(os.path.join(exp_folder, "remix_policy_rewards.csv"))
        rewards = res.iloc[:MAX_ROWS,0]
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        result_dict[exp_name] = {"mean": rewards_mean, "std": rewards_std}

    order = " & ".join([f'{game}_s42_re_pr-{pr_str}_OCAtariinput_{i}l-v3' for pr_str in ["nop","ext"] for i in [2,1]])
    eclaire_oc_groundtruth = " & ".join([f"{result_dict[f'{game}_s42_re_pr-{pr_str}_OCAtariinput_{i}l-v3']['mean']:.1f}\pm\scriptsize{{{result_dict[f'{game}_s42_re_pr-{pr_str}_OCAtariinput_{i}l-v3']['std']:.1f}}}" for pr_str in ["nop","ext"] for i in [2,1]])
    eclaire_oc_SPACE = " & ".join([f"{result_dict[f'{game}_s42_re_pr-{pr_str}_SPACEinput_{i}l-v3']['mean']:.1f}\pm\scriptsize{{{result_dict[f'{game}_s42_re_pr-{pr_str}_SPACEinput_{i}l-v3']['std']:.1f}}}" for pr_str in ["nop","ext"] for i in [2,1] ])
    print(order)
    print(eclaire_oc_groundtruth)
    print(eclaire_oc_SPACE)