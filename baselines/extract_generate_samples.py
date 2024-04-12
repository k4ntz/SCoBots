import argparse
import gymnasium as gym
import numpy as np
import time
from scobi import Environment
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import  WarpFrame
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from pathlib import Path
from typing import Callable
from rtpt import RTPT
from collections import deque
import matplotlib.pyplot as plt



def flist(l):
    return ["%.2f" % e for e in l]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="checkpoint folder containing 'best_model.zip' and 'best_vecnormalize.pkl'")
    parser.add_argument("-e", "--episodes", type=int, required=False, help="number of episodes to generate samples from")
    opts = parser.parse_args()
    
    #default values
    prune = False
    pruned_ff_name = None
    episodes = 5
    focus_dir = "focusfiles"
    
    checkpoint_name = opts.input #"Asterix_s0_re_pr"
    checkpoint_options = checkpoint_name.split("_")
    if len(checkpoint_options) == 3:
        print("unpruned")
    elif len(checkpoint_options) == 4:
        print("pruned")
        prune = True
    else:
        print("error")
    env, seed = checkpoint_options[0], checkpoint_options[1][1:]
    
    if opts.episodes:
        episodes = opts.episodes

    # print([env, seed, episodes])
    env_str = "ALE/" + env +"-v5"
    game_id = env_str.split("/")[-1].lower().split("-")[0]
    if prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"

    checkpoint_str = "best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("baselines_extract_input", checkpoint_name, checkpoint_str)
    vecnorm_path = Path("baselines_extract_input",  checkpoint_name, vecnorm_str)
    output_path = Path("baselines_extract_output", checkpoint_name)
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = output_path / "obs.npy"
        
    EVAL_ENV_SEED = 84
    env = Environment(env_str,
                      focus_dir=focus_dir,
                      focus_file=pruned_ff_name)

    _, _ = env.reset(seed=EVAL_ENV_SEED)
    dummy_vecenv = DummyVecEnv([lambda :  env])
    env = VecNormalize.load(vecnorm_path, dummy_vecenv)
    env.training = False
    env.norm_reward = False
    model = PPO.load(model_path)
    current_episode = 0
    rewards = []
    steps = []
    current_rew = 0
    current_step = 0
    obs = env.reset()

    out_array = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        out_array.append(obs)
        current_rew += reward
        current_step += 1
        if done:
            current_episode += 1
            if type(current_rew) == np.ndarray:
                current_rew = current_rew[0]
            rewards.append(current_rew)
            steps.append(current_step)
            current_rew = 0
            current_step = 0
            obs = env.reset()
        if current_episode == episodes:
            print(rewards)
            print(f"rewards: {flist(rewards)} | mean: {np.mean(rewards):.2f} \n steps: {flist(steps)} | mean: {np.mean(steps):.2f}")
            outfile.unlink(missing_ok=True)
            np.save(outfile, out_array)
            break
    #print(scores)

if __name__ == '__main__':
    main()