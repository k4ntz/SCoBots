import subprocess
import os
import numpy as np
from collections import defaultdict
# load a .npz file and return the data
def load_npz(file):
    data = np.load(file)
    # print the keys
    return data

# res = load_npz("baselines_checkpoints/Pong_s42_re_pr-ext_SPACEinput_1l-v3/evaluations.npz")

games = ["Boxing", "Pong"]

for game in games:
    result_dict = defaultdict(lambda: defaultdict(lambda: np.nan))
    for folder in os.listdir("baselines_checkpoints"):
        if not os.path.isdir(os.path.join("baselines_checkpoints", folder)):
            continue
        if not "post_training_eval.npz" in os.listdir(os.path.join("baselines_checkpoints", folder)):
            continue
        if not game in folder:
            continue
        #assert "post_training_eval.npz" in os.listdir(os.path.join("baselines_checkpoints", folder)), f"post_training_eval.npz not found in {folder} folder"
        res = load_npz(os.path.join("baselines_checkpoints", folder, "post_training_eval.npz"))
        rewards = res["rewards"]
        steps = res["steps"]
        steps_mean = np.mean(steps)
        rewards_mean = np.mean(rewards)
        steps_std = np.std(steps)
        rewards_std = np.std(rewards)
        result_dict[folder] = {"mean": rewards_mean, "std": rewards_std}

# oc_groundtruth = f""" {result_dict["Pong_s42_re_pr-nop_OCAtariinput_2l-v3"]["mean"]:.1f} & {result_dict["Pong_s42_re_pr-nop_OCAtariinput_1l-v3"]["mean"]:.1f} & {result_dict["Pong_s42_re_pr-ext_OCAtariinput_2l-v3"]["mean"]:.1f} & {result_dict["Pong_s42_re_pr-ext_OCAtariinput_1l-v3"]["mean"]:.1f}"""
# oc_space = f""" {result_dict["Pong_s42_re_pr-nop_SPACEinput_2l-v3"]["mean"]:.1f} & {result_dict["Pong_s42_re_pr-nop_SPACEinput_1l-v3"]["mean"]:.1f} & {result_dict["Pong_s42_re_pr-ext_SPACEinput_2l-v3"]["mean"]:.1f} & {result_dict["Pong_s42_re_pr-ext_SPACEinput_1l-v3"]["mean"]:.1f}"""
# oc_traingroundtruth_evalonSPACE =None

    order = " & ".join([f'{game}_s42_re_pr-{pr_str}_OCAtariinput_{i}l-v3' for pr_str in ["nop","ext"] for i in [2,1]])
    oc_groundtruth = " & ".join([f"{result_dict[f'{game}_s42_re_pr-{pr_str}_OCAtariinput_{i}l-v3']['mean']:.1f}\pm\scriptsize{{{result_dict[f'{game}_s42_re_pr-{pr_str}_OCAtariinput_{i}l-v3']['std']:.1f}}}" for pr_str in ["nop","ext"] for i in [2,1]])
    oc_space = " & ".join([f"{result_dict[f'{game}_s42_re_pr-{pr_str}_SPACEinput_{i}l-v3']['mean']:.1f}\pm\scriptsize{{{result_dict[f'{game}_s42_re_pr-{pr_str}_SPACEinput_{i}l-v3']['std']:.1f}}}" for pr_str in ["nop","ext"] for i in [2,1] ])
    print(order)
    print(oc_groundtruth)
    print(oc_space)