#!/usr/bin/env python
'''
VIPER training extraction script using denormalized data
This script is used to extract VIPER decision trees from trained models, using env.get_original_obs() to get denormalized data for training
'''

import os
import argparse
from pathlib import Path
import time
import yaml
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from rtpt import RTPT

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scobi import Environment
from utils.viper_denormalized import DenormalizedVIPER

EVAL_ENV_SEED = 84

def create_environment(game_name, focus_file=None, focus_dir="paper_experiments/focusfiles"):
    """create game environment"""
    env_name = "ALE/" + game_name + "-v5" if game_name != "CarRacing" else "CarRacing-v2"
    
    if focus_file is None:
        raise ValueError("focus_file must be specified")
    else:
        focus_dir_path = Path(focus_dir)
        if not focus_dir_path.exists():
            raise ValueError(f"focus_dir '{focus_dir}' not found")
            
        env = Environment(
            env_name, 
            focus_dir=focus_dir, 
            focus_file=focus_file.name if isinstance(focus_file, Path) else focus_file,
            hide_properties=False, 
            draw_features=True, 
            reward=0
        )
    
    # create vectorized environment
    dummy_vec_env = DummyVecEnv([lambda: env])
    return dummy_vec_env

def load_model_and_vecnormalize(model_path, vec_env, vecnorm_path=None):
    """load model and vectorized environment"""
    # check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file '{model_path}' not found")
    
    # load model: if it is a file, load it directly, if it is a directory, find best_model.zip
    if os.path.isfile(model_path):
        model = PPO.load(model_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"successfully loaded model from {model_path}")
    elif os.path.isdir(model_path):
        best_model_path = os.path.join(model_path, "best_model.zip")
        if os.path.exists(best_model_path):
            model = PPO.load(best_model_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"successfully loaded model from {best_model_path}")
        else:
            raise FileNotFoundError(f"best_model.zip not found in directory '{model_path}'")
    else:
        raise FileNotFoundError(f"model path '{model_path}' is neither a file nor a directory")
    
    # if vecnorm_path exists, load it
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False  # stop updating running mean/variance
        vec_env.norm_reward = False  # do not normalize the reward
        print(f"successfully loaded VecNormalize from {vecnorm_path}")
    else:
        # no VecNormalize file found, cannot continue
        raise FileNotFoundError("VecNormalize file not found, cannot continue. Please provide a valid vecnorm_path parameter.")
    
    # set the seed
    vec_env.seed = EVAL_ENV_SEED
    
    return model, vec_env

def extract_viper_tree(env, model, output_dir, max_depth=3, iterations=20, data_per_iter=30000):
    """extract VIPER decision tree"""
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # create decision tree
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    
    # create RTPT instance to track progress
    rtpt = RTPT(name_initials="VIPER", experiment_name="DenormalizedVIPER", max_iterations=iterations)
    rtpt.start()
    
    print("using denormalized data to extract VIPER decision tree...")
    viper = DenormalizedVIPER(
        model, 
        dt, 
        env, 
        data_per_iter=data_per_iter, 
        rtpt=rtpt
    )
    
    # start extraction
    start_time = time.time()
    viper.imitate(iterations)
    print(f"decision tree extraction completed, time taken: {time.time() - start_time:.2f} seconds")
    
    # save the best tree
    viper.save_best_tree(output_path)
    print(f"best decision tree saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='extract VIPER decision tree from trained model (using denormalized data)')
    parser.add_argument('-g', '--game', type=str, required=True, help='game name, e.g. Pong')
    parser.add_argument('-m', '--model', type=str, required=True, help='model path')
    parser.add_argument('-v', '--vecnorm', type=str, help='VecNormalize path (usually best_vecnormalize.pkl)')
    parser.add_argument('-o', '--output', type=str, required=True, help='output directory')
    parser.add_argument('-d', '--depth', type=int, default=3, help='decision tree maximum depth (default: 3)')
    parser.add_argument('-i', '--iterations', type=int, default=20, help='number of iterations (default: 20)')
    parser.add_argument('-s', '--samples', type=int, default=30000, help='number of samples per iteration (default: 30000)')
    parser.add_argument('-f', '--focus', type=str, help='focus file name')
    parser.add_argument('--focus_dir', type=str, default='paper_experiments/focusfiles', help='focus file directory (default: paper_experiments/focusfiles)')
    
    args = parser.parse_args()
    
    # if the model path is provided but the VecNormalize path is not, try to automatically find it
    if args.vecnorm is None and args.model:
        possible_vecnorm = os.path.join(args.model, "best_vecnormalize.pkl")
        if os.path.exists(possible_vecnorm):
            args.vecnorm = possible_vecnorm
            print(f"automatically detected VecNormalize file: {args.vecnorm}")
    
    # create environment
    focus_file = args.focus
    if focus_file:
        focus_file = Path(focus_file) if not '/' in focus_file else focus_file
        print(f"using focus file {focus_file}")
    
    # create vectorized environment
    vec_env = create_environment(args.game, focus_file, args.focus_dir)
    
    # load model and VecNormalize
    model, vec_env = load_model_and_vecnormalize(args.model, vec_env, args.vecnorm)
    
    # extract decision tree
    print(f"starting VIPER decision tree extraction using denormalized data, max depth={args.depth}, iterations={args.iterations}...")
    extract_viper_tree(
        env=vec_env,
        model=model,
        output_dir=args.output,
        max_depth=args.depth,
        iterations=args.iterations,
        data_per_iter=args.samples
    )
    
    print(f"VIPER decision tree extraction completed! best decision tree saved to {args.output}")
    
    # close environment
    vec_env.close()

if __name__ == "__main__":
    main() 