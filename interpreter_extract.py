from pathlib import Path

import argparse
from rtpt import RTPT
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from scobi import Environment

from utils.interpreter import Interpreter
from utils.policies import ObliqueDTPolicy, SB3Policy
from pickle import dump

EVAL_ENV_SEED = 84


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="checkpoint folder name containing 'best_model.zip' and 'best_vecnormalize.pkl'")
    parser.add_argument("-r", "--rule_extraction", type=str, required=True, choices=["interpreter"], default="interpreter", help="rule extraction to use.")
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

    # list all files that match the checkpoint name
    import os
    a = os.listdir("resources/checkpoints")
    a = list(filter(lambda x: checkpoint_name in x, a))
    num_exps = len(a)
    # choose latest
    if num_exps > 1:
        checkpoint_name += f"-n{num_exps}"
    print("using: " + checkpoint_name)
    
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
    output_path = Path("resources/interpreter_extract/extract_output", checkpoint_name + "-" + expname)
    print("Saving under " + str(output_path))
    output_path.mkdir(parents=True, exist_ok=True)

    env = Environment(env_str,
                      focus_dir=focus_dir,
                      focus_file=pruned_ff_name)
    _, _ = env.reset(seed=EVAL_ENV_SEED)

    # Original SB3 Model Eval and Trainset Generation
    model = PPO.load(model_path, device="cuda:0")
    sb3_model_wrapped = SB3Policy(model)
    dummy_vecenv = DummyVecEnv([lambda :  env])
    vec_env = VecNormalize.load(vecnorm_path, dummy_vecenv)
    vec_env.seed = EVAL_ENV_SEED
    vec_env.training = False
    vec_env.norm_reward = False

    if rule_extract == "interpreter":
        MAX_DEPTH = 7
        MAX_LEAVES = 100
        NB_TIMESTEPS = 5e4
        DATA_PER_ITER = 5000

        clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAVES)
        learner = ObliqueDTPolicy(clf, vec_env)
        interpreter = Interpreter(sb3_model_wrapped, learner, vec_env, data_per_iter=DATA_PER_ITER)
        interpreter.fit(NB_TIMESTEPS)
        print("Saving best tree with reward: " + str(interpreter.max_tree_reward))
        with open(output_path / "tree.interpreter", "wb") as f:
            dump(interpreter._policy, f)
        print("Done!")

if __name__ == "__main__":
    main()