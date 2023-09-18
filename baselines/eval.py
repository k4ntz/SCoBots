import argparse
import gymnasium as gym
import numpy as np
from scobi import Environment
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback, CallbackList, EvalCallback
from pathlib import Path
from typing import Callable
from rtpt import RTPT
from collections import deque
import matplotlib.pyplot as plt

def flist(l):
    return ["%.2f" % e for e in l]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-t", "--times", type=int, required=True,
                        help="number of episodes to eval")
    parser.add_argument("-r", "--reward", type=str, required=False, choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, required=False, choices=["default", "external"], 
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'baselines_focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-e", "--exclude_properties", action="store_true", help="exclude properties from feature vector")
    parser.add_argument("--rgb", action="store_true", help="rgb observation space")
    parser.add_argument("-l", "--liveplot", action="store_true", help="show liveplot")
    opts = parser.parse_args()

    env_str = "ALE/" + opts.game +"-v5"
    settings_str = ""
    pruned_ff_name = None
    focus_dir = "focusfiles"
    hide_properties = False
    
    reward_mode = 0
    if opts.reward == "env":
        settings_str += "_re"
    if opts.reward == "human":
        settings_str += "_rh"
        reward_mode = 1
    if opts.reward == "mixed":
        settings_str += "_rm"
        reward_mode = 2

    game_id = env_str.split("/")[-1].lower().split("-")[0]

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
    if opts.prune == "default":
        settings_str += "_pr-def"
    if opts.prune == "external":
        settings_str += "_pr-ext"
        focus_dir = "baselines_focusfiles"
    if opts.exclude_properties:
        settings_str += '_ep'
        hide_properties = True

    #override setting str if rgb
    if opts.rgb:
        settings_str = "-rgb"
    exp_name = opts.game + "_s" + str(opts.seed) + settings_str
    checkpoint_str = "model_20000000_steps" #"best_model"
    model_path = Path("baselines_checkpoints", exp_name, checkpoint_str)


    EVAL_ENV_SEED = 0
    env = Environment(env_str, 
                        focus_file=pruned_ff_name, 
                        hide_properties=hide_properties, 
                        draw_features=False, # implement feature attribution
                        reward=reward_mode) #env reward only for evaluation

    obs, info = env.reset(seed=EVAL_ENV_SEED)
    #print(obs)
    model = PPO.load(model_path)
    fps = 30
    frame_delta = 1.0 / fps
    current_episode = 0
    rewards = []
    steps = []
    current_rew = 0
    current_step = 0
    img = plt.imshow(env.original_obs)
    while True:
        action, _ = model.predict(obs)
        obs, reward, term, trunc, info = env.step(action)
        #if reward > 20:
        #print([env.original_reward, reward])
        current_rew += env.original_reward
        current_step += 1
        # TODO: implement feature attribution here
        if opts.liveplot:
            img.set_data(env.original_obs)
            plt.pause(frame_delta)
        if term or trunc:
            current_episode += 1
            rewards.append(current_rew)
            steps.append(current_step)
            current_rew = 0
            current_step = 0
            obs, _ = env.reset()
        if current_episode == opts.times:
            print(f"rewards: {flist(rewards)} | mean: {np.mean(rewards):.2f} \n steps: {flist(steps)} | mean: {np.mean(steps):.2f}")
            break


if __name__ == '__main__':
    main()