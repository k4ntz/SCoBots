import argparse
import gymnasium as gym
import numpy as np
import torch
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
import os
from experiments.my_normalizer import save_normalizer
from .train import get_settings_str

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
    parser.add_argument("-r", "--reward", type=str, default="env", choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, default= "no_prune", choices=["no_prune", "default", "external"], 
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'baselines_focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-e", "--exclude_properties", action="store_true", help="exclude properties from feature vector")
    parser.add_argument("--rgbv4", action="store_true", help="rgb observation space")
    parser.add_argument("--rgbv5", action="store_true", help="rgb observation space")
    parser.add_argument("-l", "--liveplot", action="store_true", help="show liveplot")
    parser.add_argument("--save_model", action="store_true", help="save model")
    parser.add_argument("--num_layers", type=int, choices=[1, 2], default=2, help="number of layers for mlp policy")
    parser.add_argument("--input_data", type=str, choices=["SPACE", "OCAtari",], default="SPACE", help="input data")
    parser.add_argument("--create_obs", action="store_true", help="create observation data")
    opts = parser.parse_args()

    settings_str = get_settings_str(opts)
    print(settings_str)
    
    env_str = "ALE/" + opts.game +"-v5"
    if opts.rgbv4 and opts.rgbv5:
        print("please select only one rgb mode!")
    if opts.rgbv4:
        env_str = opts.game + "NoFrameskip-v4"
    if opts.rgbv5:
        pass
    
    
    reward_mode = 0 # for eval always use env reward

    pruned_ff_name = None
    game_id = env_str.split("/")[-1].lower().split("-")[0]
    if opts.prune in ["default", "external"]:
        pruned_ff_name = f"pruned_{game_id}.yaml"

    focus_dir = "focusfiles"
    if opts.prune == "default":
        focus_dir = "focusfiles"
    if opts.prune == "external":
        focus_dir = "baselines_focusfiles"
            

    hide_properties = False
    if opts.exclude_properties:
        hide_properties = True

    exp_name = opts.game + "_s" + str(opts.seed) + settings_str #+ "_gtdata" #"-v2"



    checkpoint_str = "best_model" # "model_5000000_steps" #"best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("baselines_checkpoints", exp_name, checkpoint_str)
    vecnorm_path = Path("baselines_checkpoints",  exp_name, vecnorm_str)
    EVAL_ENV_SEED = 100 #84
    if opts.rgbv4 or opts.rgbv5:
        env = make_vec_env(env_str, seed=EVAL_ENV_SEED, wrapper_class=WarpFrame)
    else:
        env = Environment(env_str,
                            focus_dir=focus_dir,
                            focus_file=pruned_ff_name, 
                            hide_properties=hide_properties, 
                            draw_features=True, # implement feature attribution
                            reward=reward_mode, # 0 for env reward
                            object_detector= opts.input_data,
                            ) 
        _, _ = env.reset(seed=EVAL_ENV_SEED)
        dummy_vecenv = DummyVecEnv([lambda :  env])
        #print(obs)
        env = VecNormalize.load(vecnorm_path, dummy_vecenv)
        env.training = False
        env.norm_reward = False
    model = PPO.load(model_path)

    if opts.save_model:
        folder_path = f"eclaire_{exp_name}"
        os.makedirs(folder_path, exist_ok=True)
        torch.save(model.policy.state_dict(), os.path.join(folder_path, "model.pth"))
        save_normalizer(env, os.path.join(folder_path, "normalizer.json"))
        exit()
    
    fps = 30
    sps = 20 # steps per seconds
    steps_delta = 1.0 / sps
    frame_delta = 1.0 / fps
    current_episode = 0
    rewards = []
    steps = []
    current_rew = 0
    current_step = 0
    obs = env.reset()
    if opts.rgbv4 or opts.rgbv5:
        img = plt.imshow(env.get_images()[0])
    else:
        scobi_env = env.venv.envs[0]
        img = plt.imshow(scobi_env._obj_obs)

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # save observation as image
        # plt.imsave(f"tmp/obs_obj_{current_episode:05d}_{current_step:05d}.png", env.venv.envs[0]._obj_obs) # use env.envs[0].oc_env._get_obs() instead?
        
        current_rew += reward #scobi_env.original_reward
        current_step += 1
        # TODO: implement feature attribution here
        if opts.liveplot:
            if opts.rgbv4 or opts.rgbv5:
                img.set_data(env.get_images()[0])
            else:
                img.set_data(scobi_env._obj_obs)
                time.sleep(steps_delta)
            plt.pause(frame_delta)
        if done:
            current_episode += 1
            rewards.append(current_rew)
            steps.append(current_step)
            current_rew = 0
            current_step = 0
            obs = env.reset()
        if current_episode == opts.times:
            print(f"rewards: {flist(rewards)} | mean: {np.mean(rewards):.2f} \n steps: {flist(steps)} | mean: {np.mean(steps):.2f}")
            break

if __name__ == '__main__':
    main()