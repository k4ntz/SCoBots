import time
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import utils.parser.parser
from scobi import Environment


def flist(l):
    return ["%.2f" % e for e in l]


def main():
    parser = argparse.ArgumentParser()

    exp_name, env_str, hide_properties, pruned_ff_name, variant = utils.parser.parser.parse_render(parser)

    checkpoint_str = "best_model" # "model_5000000_steps" #"best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("checkpoints", exp_name, checkpoint_str)
    vecnorm_path = Path("checkpoints",  exp_name, vecnorm_str)
    EVAL_ENV_SEED = 84
    if variant == "rgb":
        env = make_vec_env(env_str, seed=EVAL_ENV_SEED, wrapper_class=WarpFrame)
    else:
        env = Environment(env_str,
                            focus_file=pruned_ff_name,
                            hide_properties=hide_properties,
                            draw_features=True, # implement feature attribution
                            reward=0) #env reward only for evaluation

        _, _ = env.reset(seed=EVAL_ENV_SEED)
        dummy_vecenv = DummyVecEnv([lambda :  env])
        env = VecNormalize.load(vecnorm_path, dummy_vecenv)
        env.training = False
        env.norm_reward = False
    model = PPO.load(model_path)
    fps = 30
    sps = 20 # steps per seconds
    steps_delta = 1.0 / sps
    frame_delta = 1.0 / fps
    obs = env.reset()
    if variant == "rgb":
        img = plt.imshow(env.get_images()[0])
    else:
        scobi_env = env.venv.envs[0]
        img = plt.imshow(scobi_env._obj_obs)

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if variant == "rgb":
            img.set_data(env.get_images()[0])
        else:
            img.set_data(scobi_env._obj_obs)
            time.sleep(steps_delta)
        plt.pause(frame_delta)
        if done:
            obs = env.reset()

if __name__ == '__main__':
    main()