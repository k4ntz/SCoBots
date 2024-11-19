import csv
import os
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from joblib import load
from viper_extract import DTClassifierModel


import utils.parser.parser
from scobi import Environment


def flist(l):
    return ["%.2f" % e for e in l]

# Saving in a CSV file the results of the evaluation
def _save_evals(rewards, mean_rewards, mean_steps, csv_filename):

    file_exists = os.path.isfile(csv_filename)
    lowest_reward = min(rewards)
    highest_reward = max(rewards)

    # Write to CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file is new
        if not file_exists:
            writer.writerow(['lowest reward', 'highest rewards', 'mean reward', 'mean_steps'])

        # Write the current data
        writer.writerow([lowest_reward, highest_reward, mean_rewards, mean_steps])

    print(f"Data saved to {csv_filename}")

# Helper function to load from a dt and not a checkpoint directly
def _load_viper(exp_name, path_provided):
    if path_provided:
        viper_path = Path(exp_name)
        model = load(sorted(viper_path.glob("*_best.viper"))[0])
    else:
        viper_path = Path("resources/viper_extracts/extract_output", exp_name + "-extraction")
        model = load(sorted(viper_path.glob("*_best.viper"))[0])

    wrapped = DTClassifierModel(model)

    return wrapped

# Helper function ensuring that the loaded checkpoint has completed training
def _ensure_completeness(path):
    checkpoint = path / "best_model.zip"
    return checkpoint.is_file()

def main():
    parser = argparse.ArgumentParser()

    flag_dictionary = utils.parser.parser.parse_eval(parser)
    version = int(flag_dictionary["version"])
    exp_name = flag_dictionary["exp_name"]
    variant = flag_dictionary["variant"]
    env_str = flag_dictionary["env_str"]
    pruned_ff_name = flag_dictionary["pruned_ff_name"]
    hide_properties = flag_dictionary["hide_properties"]
    viper = flag_dictionary["viper"]
    progress_bar = flag_dictionary["progress"]
    time = int(flag_dictionary["times"])

    if version == 0:
        version = utils.parser.parser.get_highest_version(exp_name)

    exp_name += str(version)
    checkpoint_str = "best_model" # "model_5000000_steps" #"best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("resources/checkpoints", exp_name, checkpoint_str)
    vecnorm_path = Path("resources/checkpoints",  exp_name, vecnorm_str)
    ff_file_path = Path("resources/checkpoints", exp_name)
    if not _ensure_completeness(model_path):
        print('Training not completed!')
        print('Delete the folder ' + str(ff_file_path) + ' or complete the training process')
        return
    EVAL_ENV_SEED = 84
    if variant == "rgb":
        env = make_vec_env(env_str, seed=EVAL_ENV_SEED, wrapper_class=WarpFrame)
    else:
        env = Environment(env_str,
                          focus_dir=ff_file_path,
                          focus_file=pruned_ff_name,
                          hide_properties=hide_properties,
                          draw_features=True, # implement feature attribution
                          reward=0) #env reward only for evaluation

        _, _ = env.reset(seed=EVAL_ENV_SEED)
        dummy_vecenv = DummyVecEnv([lambda :  env])
        env = VecNormalize.load(vecnorm_path, dummy_vecenv)
        env.training = False
        env.norm_reward = False
    if viper:
        print("loading viper tree of " + exp_name)
        if isinstance(viper, str):
            model = _load_viper(viper, True)
        else:
            model = _load_viper(exp_name, False)
    else:
        model = PPO.load(model_path)

    current_episode = 0
    rewards = []
    steps = []
    current_rew = 0
    current_step = 0
    obs = env.reset()
    if variant == "rgb":
        img = plt.imshow(env.get_images()[0])
    else:
        scobi_env = env.venv.envs[0]
        img = plt.imshow(scobi_env._obj_obs)

    if progress_bar:
        with tqdm(total=time, desc="Episodes completed") as pbar:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                current_rew += reward #scobi_env.original_reward
                current_step += 1
                if done:
                    current_episode += 1
                    pbar.update(1)
                    rewards.append(current_rew)
                    steps.append(current_step)
                    current_rew = 0
                    current_step = 0
                    obs = env.reset()
                if current_episode == time:
                    print(f"rewards: {flist(rewards)} | mean: {np.mean(rewards):.2f} \n steps: {flist(steps)} | mean: {np.mean(steps):.2f}")
                    _save_evals(rewards, np.mean(rewards), np.mean(steps), "resources/checkpoints/" + exp_name + "/" + "evaluation")
                    pbar.close()
                    break
    else:
        while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                current_rew += reward #scobi_env.original_reward
                current_step += 1
                if done:
                    current_episode += 1
                    rewards.append(current_rew)
                    steps.append(current_step)
                    current_rew = 0
                    current_step = 0
                    obs = env.reset()
                if current_episode == time:
                    print(f"rewards: {flist(rewards)} | mean: {np.mean(rewards):.2f} \n steps: {flist(steps)} | mean: {np.mean(steps):.2f}")
                    _save_evals(rewards, np.mean(rewards), np.mean(steps), "resources/checkpoints/" + exp_name + "/" + "evaluation")
                    break

if __name__ == '__main__':
    main()