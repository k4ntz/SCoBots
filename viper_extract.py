from pathlib import Path

import argparse
import numpy as np
# import tensorflow as tf
from joblib import load
from rtpt import RTPT
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from scobi import Environment
from utils.viper import VIPER

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


class DTClassifierModel():
    def __init__(self, model) -> None:
        self.name = "DT Classifier Model"
        self.model = model

    def predict(self, obs, deterministic=True):
        out = self.model.predict(obs)
        return np.array(out), None


def flist(l):
    return ["%.2f" % e for e in l]


def eval_agent(model, env, episodes, obs_save_file=None, acts_save_file=None):
    current_episode = 0
    rewards = []
    steps = []
    current_rew = 0
    current_step = 0
    obs_out_array = []
    acts_out_array = []
    obs = env.reset()
    while True:
        obs_out_array.append(obs[0]) #unvec
        action, _ = model.predict(obs, deterministic=True)
        acts_out_array.append(action[0]) #unvec
        obs, reward, done, _ = env.step(action)
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
                np.save(obs_save_file, obs_out_array)
                acts_save_file.unlink(missing_ok=True)
                np.save(acts_save_file, acts_out_array)
                print(">>> Observations & Actions saved!")
            print("--------------------------------------------\n")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="checkpoint folder name containing 'best_model.zip' and 'best_vecnormalize.pkl'")
    parser.add_argument("-r", "--rule_extraction", type=str, required=True, choices=["viper"], default="viper", help="rule extraction to use.")
    parser.add_argument("-e", "--episodes", type=int, required=False, help="number of episodes to evaluate agents samples on")
    parser.add_argument("-n", "--name", type=str, required=False, help="experiment name")
    opts = parser.parse_args()

    # Default values
    path_entered = False
    prune = False
    pruned_ff_name = None
    episodes = 5
    focus_dir = "resources/focusfiles"
    expname = opts.name if opts.name else "extraction"
    rule_extract = opts.rule_extraction
    if "/" in opts.input:
        checkpoint_name = opts.input.split("/")[-1]
        path_entered = True
    else:
        checkpoint_name = opts.input
    checkpoint_options = checkpoint_name.split("_")
    if len(checkpoint_options) == 4:
        print("unpruned")
    elif len(checkpoint_options) == 5:
        print("pruned")
        prune = True
        if checkpoint_options[-1] == "pr-ext-abl":
            focus_dir = "paper_experiments/norel_focusfiles"
    else:
        print("Wrong format. Format needed: 'Asterix_seed0_reward-env_pruned' or 'Asterix_seed0_reward-env'")
    env, seed = checkpoint_options[0], checkpoint_options[1][1:]

    if opts.episodes:
        episodes = opts.episodes

    env_str = "ALE/" + env +"-v5"
    game_id = env_str.split("/")[-1].lower().split("-")[0]
    if prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"

    checkpoint_str = "best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    if path_entered:
        model_path = Path(opts.input, checkpoint_str)
        vecnorm_path = Path(opts.input, vecnorm_str)
        focus_dir = Path(opts.input)
    else:
        model_path = Path("resources/checkpoints", checkpoint_name, checkpoint_str)
        vecnorm_path = Path("resources/checkpoints",  checkpoint_name, vecnorm_str)
    print("Looking for focus file in " + str(focus_dir))
    print("Looking for model in " + str(model_path))
    output_path = Path("resources/viper_extracts/extract_output", checkpoint_name + "-" + expname)
    print("Saving under " + str(output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    obs_outfile = output_path / "obs.npy"
    acts_outfile = output_path / "acts.npy"

    env = Environment(env_str,
                      focus_dir=focus_dir,
                      focus_file=pruned_ff_name)
    _, _ = env.reset(seed=EVAL_ENV_SEED)


    # Original SB3 Model Eval and Trainset Generation
    model = PPO.load(model_path, device="cuda:0")
    sb3_model_wrapped = SB3Model(model=model)
    dummy_vecenv = DummyVecEnv([lambda :  env])
    vec_env = VecNormalize.load(vecnorm_path, dummy_vecenv)
    vec_env.seed = EVAL_ENV_SEED
    vec_env.training = False
    vec_env.norm_reward = False
    eval_agent(sb3_model_wrapped, vec_env, episodes=episodes, obs_save_file=obs_outfile, acts_save_file=acts_outfile)


    if rule_extract == "viper":
        MAX_DEPTH = 7
        NB_ITER = 25
        process_name = checkpoint_name + "_" + expname
        rtpt = RTPT(name_initials="QD", experiment_name=process_name, max_iterations=NB_ITER)
        rtpt.start()
        train_observations = np.load(obs_outfile)
        train_actions = np.load(acts_outfile)
        clf = DecisionTreeClassifier(max_depth=MAX_DEPTH)
        vip = VIPER(model, clf, vec_env, rtpt)
        vip.imitate(nb_iter=NB_ITER)
        vip.save_best_tree(output_path)
        best_viper = sorted(output_path.glob("*_best.viper"))
        if not best_viper:
            print("error")
            exit()
        dtree = load(best_viper[0]) #only one should exist
        viper_wrapped = DTClassifierModel(dtree)
        eval_agent(viper_wrapped, vec_env, episodes=episodes)
        print("Done!")
if __name__ == '__main__':
    print("reached")
    main()