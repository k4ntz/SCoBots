import time
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from utils.parser.parser import render_parser, convert_args, get_highest_version
from utils.renderer import Renderer
from scobi import Environment


def flist(l):
    return ["%.2f" % e for e in l]


def main():
    opts = render_parser()
    exp_name, env_str, hide_properties, pruned_ff_name, variant, version, normalize, hud = convert_args(opts)
    
    if version == 0:
        version = get_highest_version(exp_name)

    exp_name += version

    checkpoint_str = "best_model" # "model_5000000_steps" #"best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("resources/checkpoints", exp_name, checkpoint_str)
    vecnorm_path = Path("resources/checkpoints",  exp_name, vecnorm_str)
    EVAL_ENV_SEED = 84
    if variant == "rgb":
        env = make_vec_env(env_str, seed=EVAL_ENV_SEED, wrapper_class=WarpFrame)
    else:
        env = Environment(env_str,
                            focus_file=pruned_ff_name,
                            hide_properties=hide_properties,
                            draw_features=True, # implement feature attribution
                            reward_mode=0, #env reward only for evaluation
                            normalize=normalize,
                            hud=hud
                            )

        _, _ = env.reset(seed=EVAL_ENV_SEED)
        dummy_vecenv = DummyVecEnv([lambda :  env])
        env = VecNormalize.load(vecnorm_path, dummy_vecenv)
        env.training = False
        env.norm_reward = False
    print("Loading model from: ", model_path)
    model = PPO.load(model_path)
    obs = env.reset()
    renderer = Renderer(env, model, opts.record, opts.nb_frames)
    renderer.print_reward = opts.print_reward
    renderer.run()

if __name__ == '__main__':
    main()