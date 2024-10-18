from pathlib import Path

import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import utils.parser.parser
from utils.renderer import Renderer
from scobi import Environment


def flist(l):
    return ["%.2f" % e for e in l]


def main():
    parser = argparse.ArgumentParser()

    exp_name, env_str, hide_properties, pruned_ff_name, variant, version, normalize, hud = utils.parser.parser.parse_render(parser)
    
    if version == 0:
        version = utils.parser.parser.get_highest_version(exp_name)

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

    if variant == "rgb":
        fps = 30
        frame_delta = 1.0 / fps
        obs = env.reset()
        img = plt.imshow(env.get_images()[0])
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if variant == "rgb":
                img.set_data(env.get_images()[0])
                img = plt.imshow(env.get_images()[0])
            plt.pause(frame_delta)
            if done:
                obs = env.reset()
    else:
        renderer = Renderer(env, model)
        renderer.run()

if __name__ == '__main__':
    main()