import os
from collections import deque
from pathlib import Path
from typing import Callable
from tqdm import tqdm

import argparse
import gymnasium as gym
import numpy as np
import torch as th
import yaml
from rtpt import RTPT
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback, CallbackList, \
    EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecTransposeImage
from datetime import datetime

import utils.parser.parser
from scobi import Environment

MULTIPROCESSING_START_METHOD = "spawn" if os.name == 'nt' else "fork"  # 'nt' == Windows

class RtptCallback(BaseCallback):
    def __init__(self, exp_name, max_iter, verbose=0):
        super(RtptCallback, self).__init__(verbose)
        self.rtpt = RTPT(name_initials="AA",
            experiment_name=exp_name,
            max_iterations=max_iter)
        self.rtpt.start()

    def _on_step(self) -> bool:
        self.rtpt.step()
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, n_envs, verbose=0):
        self.n_envs = n_envs
        self.buffer = deque(maxlen=100) #ppo default stat window
        super().__init__(verbose)

    def _on_step(self) -> bool:
        ep_rewards = self.training_env.get_attr("ep_env_reward", range(self.n_envs))
        for rew in ep_rewards:
            if rew is not None:
                self.buffer.extend([rew])


    def on_rollout_end(self) -> None:
        buff_list = list(self.buffer)
        if len(buff_list) == 0:
            return
        self.logger.record("rollout/ep_env_rew_mean", np.mean(list(self.buffer)))


class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path: str, rgb=False):
        super(SaveBestModelCallback, self).__init__()
        self.save_path = save_path
        self.rgb = rgb
        self.vec_path_name = os.path.join(self.save_path, "best_vecnormalize.pkl")

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if not self.rgb:
            self.model.get_vec_normalize_env().save(self.vec_path_name)
        self.model.save(os.path.join(self.save_path, "best_model"))

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def _create_yaml(flags, location):
    data = {
        'game': flags['game'],
        'seed': flags['seed'],
        'env': flags['environments'],
        'reward': flags['reward'],
        'prune': flags['prune'],
        'exclude_properties': flags['exclude_properties'],
        'rgbv5': flags['rgb'],
        'status': 'not finished',
        'completed_steps': None,
        'creation date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(location, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

    print(f"YAML file with training values created at {location}")


def _update_yaml(location, steps, finished):
    with open(location, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    data['completed_steps'] = steps
    data['status'] = 'finished' if finished else 'not finished'
    with open(location, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

    print(f"YAML file updated with training duration at {location}")

def _get_directory(path, exp_name):
    version_counter = 1
    while True:
        versioned_dir = path / f"{exp_name}-n{version_counter}"
        if not versioned_dir.exists():
            return versioned_dir
        version_counter += 1

def main():
    parser = argparse.ArgumentParser()
    exp_name, env_str, hide_properties, pruned_ff_name, focus_dir, reward_mode, rgb_exp, seed, envs, game, rgb, reward = utils.parser.parser.parse_train(parser)

    n_envs = envs
    n_eval_envs = 4
    n_eval_episodes = 8
    eval_env_seed = (seed + 42) * 2 #different seeds for eval
    training_timestamps = 20_000_000
    checkpoint_frequency = 1_000_000
    eval_frequency = 500_000
    rtpt_frequency = 100_000
    bar_update_interval = 100_000

    log_path = _get_directory(Path("resources/training_logs"), exp_name)
    ckpt_path = _get_directory(Path("resources/checkpoints"), exp_name)
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    yaml_path = Path(ckpt_path, f"{exp_name}_training_status.yaml")
    rgb_yaml = 'used' if rgb else 'not used'
    flags = {
        'game': game,
        'seed': seed,
        'environments': envs,
        'reward': reward,
        'prune': pruned_ff_name,
        'exclude_properties': hide_properties,
        'rgb': rgb_yaml
    }
    _create_yaml(flags, yaml_path)




    def make_env(rank: int = 0, seed: int = 0, silent=False, refresh=True) -> Callable:
        def _init() -> gym.Env:
            env = Environment(env_str,
                              seed=seed + rank,
                              focus_dir=focus_dir,
                              focus_file=pruned_ff_name,
                              hide_properties=hide_properties,
                              silent=silent,
                              reward=reward_mode,
                              refresh_yaml=refresh)
            env = EpisodicLifeEnv(env=env)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init

    def make_eval_env(rank: int = 0, seed: int = 0, silent=False, refresh=True) -> Callable:
        def _init() -> gym.Env:
            env = Environment(env_str,
                              seed=seed + rank,
                              focus_dir=focus_dir,
                              focus_file=pruned_ff_name,
                              hide_properties=hide_properties,
                              silent=silent,
                              reward=0, #always env reward for eval
                              refresh_yaml=refresh)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # preprocessing based on atari wrapper of the openai baseline implementation (https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py)
    if rgb:
        # NoopResetEnv not required, because v5 has sticky actions, and also frame_skip=5, so also not required to set in wrapper (1 means no frameskip). no reward clipping, because scobots dont clip as well
        train_wrapper_params = {"noop_max" : 0, "frame_skip" : 1, "screen_size": 84, "terminal_on_life_loss": True, "clip_reward" : False} # remaining values are part of AtariWrapper
        train_env = make_vec_env(env_str, n_envs=n_envs, seed=seed,  wrapper_class=AtariWrapper, wrapper_kwargs=train_wrapper_params, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        train_env = VecTransposeImage(train_env) #required for PyTorch convolution layers.
        # disable EpisodicLifeEnv, ClipRewardEnv for evaluation
        eval_wrapper_params = {"noop_max" : 0, "frame_skip" : 1, "screen_size": 84, "terminal_on_life_loss": False, "clip_reward" : False} # remaining values are part of AtariWrapper
        eval_env = make_vec_env(env_str, n_envs=n_eval_envs, seed=eval_env_seed, wrapper_class=AtariWrapper, wrapper_kwargs=eval_wrapper_params, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        eval_env = VecTransposeImage(eval_env) #required for PyTorch convolution layers.
    else:
        # check if compatible gym env
        monitor = make_env()()
        check_env(monitor.env)
        del monitor
        # silent init and dont refresh default yaml file because it causes spam and issues with multiprocessing
        eval_env = VecNormalize(SubprocVecEnv([make_eval_env(rank=i, seed=eval_env_seed, silent=True, refresh=False) for i in range(n_eval_envs)], start_method=MULTIPROCESSING_START_METHOD), norm_reward=False, training=False)
        train_env = VecNormalize(SubprocVecEnv([make_env(rank=i, seed=seed, silent=True, refresh=False) for i in range(n_envs)], start_method=MULTIPROCESSING_START_METHOD), norm_reward=False)

    rtpt_iters = training_timestamps // rtpt_frequency
    save_bm = SaveBestModelCallback(ckpt_path, rgb=rgb_exp)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=save_bm,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=str(ckpt_path),
        log_path=str(ckpt_path),
        eval_freq=max(eval_frequency // n_envs, 1),
        deterministic=True,
        render=False)

    checkpoint_callback = CheckpointCallback(
        save_freq= max(checkpoint_frequency // n_envs, 1),
        save_path=str(os.path.join(ckpt_path,'training_checkpoints')),
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True)

    rtpt_callback = RtptCallback(
        exp_name=exp_name,
        max_iter=rtpt_iters)

    n_callback = EveryNTimesteps(
        n_steps=rtpt_frequency,
        callback=rtpt_callback)

    progress_bar_callback = ProgressBarCallback(total_timesteps=training_timestamps)

    tb_callback = TensorboardCallback(n_envs=n_envs)
    cbl = [checkpoint_callback, eval_callback, n_callback, tb_callback, progress_bar_callback]
    if rgb_exp: #remove tb callback if rgb
        cbl = cbl[:-1]
    cb_list = CallbackList(cbl)
    new_logger = configure(str(log_path), ["tensorboard"])

    if rgb_exp:
        # pkwargs = dict(activation_fn=th.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
        # was mentioned in ppo paper, but most likely not used, use Nature DQN:
        # (https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py#L6) instead
        # this is implemented in sb3 as 'CnnPolicy'
        policy_str = "CnnPolicy"
        adam_step_size = 0.00025
        clipping_eps = 0.1
        norm_adv = True
        model = PPO(
            policy=policy_str,
            n_steps=128,
            learning_rate=linear_schedule(adam_step_size),
            n_epochs=3,
            batch_size=32*8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(clipping_eps),
            normalize_advantage=norm_adv,
            vf_coef=1,
            ent_coef=0.01,
            env=train_env,
            #policy_kwargs=pkwargs,
            verbose=1)
    else:
        policy_str = "MlpPolicy"
        pkwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[64, 64], vf=[64, 64]))
        adam_step_size = 0.001
        if game in ["Bowling", "Tennis"]:
            adam_step_size = 0.00025
        clipping_eps = 0.1
        model = PPO(
            policy_str,
            n_steps=2048,
            learning_rate=linear_schedule(adam_step_size),
            n_epochs=3,
            batch_size=32*8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(clipping_eps),
            vf_coef=1,
            ent_coef=0.01,
            env=train_env,
            policy_kwargs=pkwargs,
            verbose=1)
    model.set_logger(new_logger)
    print(model.policy)
    print(f"Experiment name: {exp_name}")
    print(f"Started {type(model).__name__} training with {n_envs} actors and {n_eval_envs} evaluators...")  

    model.learn(total_timesteps=training_timestamps, callback=cb_list)

    _update_yaml(yaml_path, model.num_timesteps, True)

if __name__ == '__main__':
    main()