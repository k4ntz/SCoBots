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
from remix import eclaire, deep_red_c5
from remix.rules.ruleset import Ruleset
import torch
import numpy as np
import tensorflow as tf

EVAL_ENV_SEED = 84

class SB3Model():
    def __init__(self, model) -> None:
        self.name = "Original SB3 Model"
        self.model = model

    def predict(self, obs, deterministic):
        return self.model.predict(obs, deterministic) #vecenv output eg. (array([2]), True)


class KerasModel():
    def __init__(self, model) -> None:
        self.name = "Translated Keras Model"
        self.model = model

    def predict(self, obs, deterministic=True):
        out = self.model(obs)
        idx = tf.argmax(out[0])
        return np.array([idx]), None #increase dim to match vecenv used


class RemixModel():
    def __init__(self, model) -> None:
        self.name = "Extracted Ruleset Model"
        self.model = model

    def predict(self, obs, deterministic=True):
        out = self.model.predict(obs)
        return np.array(out), None


def flist(l):
    return ["%.2f" % e for e in l]


def eval_agent(model, env, episodes, obs_save_file=None):
    current_episode = 0
    rewards = []
    steps = []
    current_rew = 0
    current_step = 0
    out_array = []
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        out_array.append(obs[0]) #unvec
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
            print("--------------------------------------------\n"+model.name)
            print(f"rewards: {flist(rewards)} | mean: {np.mean(rewards):.2f} \nsteps: {flist(steps)} | mean: {np.mean(steps):.2f}")
            if obs_save_file:
                obs_save_file.unlink(missing_ok=True)
                np.save(obs_save_file, out_array)
                print(">>> Observations saved!")
            print("--------------------------------------------\n")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="checkpoint folder containing 'best_model.zip' and 'best_vecnormalize.pkl'")
    parser.add_argument("-e", "--episodes", type=int, required=False, help="number of episodes to generate samples from")
    parser.add_argument("-n", "--name", type=str, required=False, help="experiment name")
    opts = parser.parse_args()
    
    # Default values
    prune = False
    pruned_ff_name = None
    episodes = 5
    focus_dir = "focusfiles"
    expname = opts.name if opts.name else "experiment"
    checkpoint_name = opts.input #"Asterix_s0_re_pr"
    checkpoint_options = checkpoint_name.split("_")
    if len(checkpoint_options) == 3:
        print("unpruned")
    elif len(checkpoint_options) == 4:
        print("pruned")
        prune = True
    else:
        print("Wrong format. Format needed: 'Asterix_s0_re_pr' or 'Asterix_s0_re'. seed0, re:reward from env, pr:pruned")
    env, seed = checkpoint_options[0], checkpoint_options[1][1:]
    
    if opts.episodes:
        episodes = opts.episodes

    env_str = "ALE/" + env +"-v5"
    game_id = env_str.split("/")[-1].lower().split("-")[0]
    if prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"

    checkpoint_str = "best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("baselines_extract_input", checkpoint_name, checkpoint_str)
    vecnorm_path = Path("baselines_extract_input",  checkpoint_name, vecnorm_str)
    output_path = Path("baselines_extract_output", checkpoint_name + "-" + expname)
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = output_path / "obs.npy"
        

    env = Environment(env_str,
                      focus_dir=focus_dir,
                      focus_file=pruned_ff_name)
    _, _ = env.reset(seed=EVAL_ENV_SEED)


    # Original SB3 Model Eval and Trainset Generation
    model = PPO.load(model_path)
    sb3_model_wrapped = SB3Model(model=model)
    dummy_vecenv = DummyVecEnv([lambda :  env])
    vec_env = VecNormalize.load(vecnorm_path, dummy_vecenv)
    vec_env.seed = EVAL_ENV_SEED
    vec_env.training = False
    vec_env.norm_reward = False
    eval_agent(sb3_model_wrapped, vec_env, episodes=episodes, obs_save_file=outfile)


    # Translation to Keras and Model Eval 
    fnames = env.get_vector_entry_descriptions()
    actions = env.action_space_description
    input_size = len(fnames)
    output_size = len(actions)
    pi_hidden_sizes = model.policy.net_arch["pi"]
    torch_act_f_name = str(model.policy.activation_fn).split(".")[-1][:-2]
    keras_act_f = torch_act_f_name.lower() #works for relu, didnt test others

    #hardcoded to 2 hidden layers for now. should work for arbitrary sizes
    pi_weight_keys = ['mlp_extractor.policy_net.0.weight', 
               'mlp_extractor.policy_net.0.bias',
               'mlp_extractor.policy_net.2.weight',
               'mlp_extractor.policy_net.2.bias',
               'action_net.weight',
               'action_net.bias']

    weights = model.policy.state_dict()
    keras_model = tf.keras.Sequential()
    keras_model.add(tf.keras.Input(shape=(input_size,)))
    keras_model.add(tf.keras.layers.Activation("linear"))
    keras_model.add(tf.keras.layers.Dense(pi_hidden_sizes[0], activation=keras_act_f))
    keras_model.add(tf.keras.layers.Dense(pi_hidden_sizes[1], activation=keras_act_f))
    keras_model.add(tf.keras.layers.Dense(output_size))
    keras_model.add(tf.keras.layers.Softmax())
    keras_model.set_weights([weights[key].cpu().numpy().T for key in pi_weight_keys])
    #keras_model.summary()

    keras_model_wrapped = KerasModel(keras_model)
    eval_agent(keras_model_wrapped, vec_env, episodes=episodes)
    

    # Rule extraction via ECLAIRE
    ruleset_fname = checkpoint_name + ".rules"
    trainset = np.load(outfile)
    input = tf.convert_to_tensor(trainset.astype(np.float32))
    clean_fnames = [s.replace(' ', '') for s in fnames] # need to remove whitespaces in names otherwise remix gets mad
    ruleset = eclaire.extract_rules(keras_model, input, feature_names=clean_fnames, output_class_names=actions)
    ruleset.to_file(output_path / ruleset_fname)
    print(f"Ruleset saved as '%s' !" % ruleset_fname)

    # Eval ruleset
    ruleset = Ruleset().from_file(output_path / ruleset_fname)
    ruleset_model_wrapped = RemixModel(ruleset)
    eval_agent(ruleset_model_wrapped, vec_env, episodes=episodes)
    print("Done!")

if __name__ == '__main__':
    main()