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



def main():



    # TODO: make reward_shaping with unpruned mode work, change in scobi, doesnt make sense to have 
    # "interactive" and "non interactive" modes
    # TODO: add "exclude concepts"
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-t", "--times", type=int, required=True,
                        help="number of episodes to eval")
    parser.add_argument("-r", "--reward", type=str, required=True, choices=["env", "human", "mixed"],
                        help="reward mode")
    parser.add_argument("-p", "--prune", action="store_true", help="use pruned focusfile")
    parser.add_argument("-e", "--exclude_properties", action="store_true", help="exclude properties from feature vector")
    parser.add_argument("-l", "--liveplot", action="store_true", help="show liveplot")

    opts = parser.parse_args()

    env_str = "ALE/" + opts.game +"-v5"
    settings_str = ""
    pruned_ff_name = None
    hide_properties = False
    if opts.reward == "env":
        settings_str += "_re"
    if opts.reward == "human":
        settings_str += "_rh"
    if opts.reward == "mixed":
        settings_str += "_rm"
    if opts.prune:
        settings_str += "_pr"
        game_id = env_str.split("/")[-1].lower().split("-")[0] 
        pruned_ff_name = f"pruned_{game_id}.yaml"
    if opts.exclude_properties:
        settings_str += '_ep'
        hide_properties = True

    exp_id_str = opts.game + "_s" + str(opts.seed) + settings_str
    model_path = Path("baseline_checkpoints", exp_id_str, "best_model")


    EVAL_ENV_SEED = 0
    env = Environment(env_str,
                      focus_file=pruned_ff_name,
                      hide_properties=hide_properties,
                      draw_features=False,  # implement feature attribution
                      reward_mode=0) #env reward only for evaluation

    obs, info = env.reset(seed=EVAL_ENV_SEED)
    print(obs)
    model = PPO.load(model_path)
    fps = 30
    frame_delta = 1.0 / fps
    current_episode = 0
    rewards = []
    current_rew = 0
    img = plt.imshow(env.original_obs)
    while True:
        action, _ = model.predict(obs)
        obs, reward, term, trunc, info = env.step(action)
        current_rew += reward
        # TODO: implement feature attribution here
        if opts.liveplot:
            img.set_data(env.original_obs)
            plt.pause(frame_delta)
        if term or trunc:
            current_episode += 1
            rewards.append(current_rew)
            current_rew = 0
            obs, _ = env.reset()
        if current_episode == opts.times:
            print(f"rewards: {rewards} | mean: {np.mean(rewards)}")
            break


if __name__ == '__main__':
    main()