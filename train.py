import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

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

import utils.parser.parser
from scobi import Environment

os.environ['OMP_NUM_THREADS'] = '4'  # Limit threads for OpenMP
os.environ['MKL_NUM_THREADS'] = '4'  # Limit threads for MKL (used by NumPy)
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Limit threads for OpenBLAS

# MULTIPROCESSING_START_METHOD = "spawn" if os.name == 'nt' else "fork"  # 'nt' == Windows
# MULTIPROCESSING_START_METHOD = "spawn" if os.name == 'nt' else "forkserver"  # 'nt' == Windows

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
        self.ep_env_buffer = deque(maxlen=100) #ppo default stat window
        self.ep_rew_shape_buffer = deque(maxlen=100)
        super().__init__(verbose)

    def _on_step(self) -> bool:
        ep_env_rewards = self.training_env.get_attr("ep_env_reward", range(self.n_envs))
        ep_rew_shape_rewards = self.training_env.get_attr("ep_rew_shape_reward", range(self.n_envs))
        for rew in ep_env_rewards:
            if rew is not None:
                self.ep_env_buffer.extend([rew])
        for rew in ep_rew_shape_rewards:
            if rew is not None:
                self.ep_rew_shape_buffer.extend([rew])


    def on_rollout_end(self) -> None:
        buff_env_list = list(self.ep_env_buffer)
        buff_rew_shape_list = list(self.ep_rew_shape_buffer)
        if len(buff_env_list) != 0:
            self.logger.record("rollout/ep_env_rew_mean", np.mean(list(self.ep_env_buffer)))
        if len(buff_rew_shape_list) != 0:
            self.logger.record("rollout/ep_rew_shape_0", np.mean(list(self.ep_rew_shape_buffer), axis=0)[0])
            self.logger.record("rollout/ep_rew_shape_1", np.mean(list(self.ep_rew_shape_buffer), axis=0)[1])
            self.logger.record("rollout/ep_rew_shape_2", np.mean(list(self.ep_rew_shape_buffer), axis=0)[2])



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

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_exponential_schedule(initial_value: float, half_life_period: float = 0.25) -> Callable[[float], float]:
    """It holds exponential(half_life_period) = 0.5. If half_life_period == 0.25, then
    exponential(0) ~= 0.06"""
    assert 0 < half_life_period < 1

    def exponential(progress_remaining: float) -> float:
        return initial_value * np.exp((1 - progress_remaining) * np.log(0.5) / half_life_period)

    return exponential

def _create_yaml(flags, location):
    data = {
        'game': flags['game'],
        'seed': flags['seed'],
        'env': flags['environments'],
        'reward': flags['reward'],
        'prune': flags['prune'],
        'exclude_properties': flags['exclude_properties'],
        'rgbv5': flags['rgb'],
        'normalize': flags['normalize'],
        'status': 'not finished',
        'completed_steps': None,
        'creation date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    comments = {
        'game': "# The game trained on",
        'seed': "# Initialised on seed number",
        'env': "# Number of environments",
        'reward': "# Reward type: env, human, or mixed",
        'prune': "# Used a custom focus file using pruning",
        'exclude_properties': "# Excluding some properties",
        'rgbv5': "# Was the RGB space used",
        'status': "# Training status",
        'completed_steps': "# Steps completed in training",
        'creation date': "# File creation date"
    }

    yaml_content = yaml.dump(data, default_flow_style=False)

    commented_yaml_lines = []
    for line in yaml_content.splitlines():
        key = line.split(":")[0].strip()
        if key in comments:
            commented_yaml_lines.append(comments[key])
        commented_yaml_lines.append(line)

    commented_yaml_content = "\n".join(commented_yaml_lines)

    with open(location, 'w') as yaml_file:
        yaml_file.write(commented_yaml_content)

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
    version_counter = 2
    if not (path / f"{exp_name}").exists():
        return path / f"{exp_name}"
    while True:
        versioned_dir = path / f"{exp_name}-n{version_counter}"
        if not versioned_dir.exists():
            return versioned_dir
        version_counter += 1

def main():
    flags_dictionary = utils.parser.parser.parse_train()

    exp_name = flags_dictionary["exp_name"]
    n_envs = int(flags_dictionary["environments"])
    continue_ckpt = flags_dictionary["continue_ckpt"]
    n_eval_envs = 4
    n_eval_episodes = 8
    eval_env_seed = (int(flags_dictionary["seed"]) + 42) * 2 #different seeds for eval
    training_timestamps = 20_000_000
    checkpoint_frequency = 1_000_000
    # eval_frequency = 500_000
    eval_frequency = 250_000
    rtpt_frequency = 100_000

    log_path = _get_directory(Path("resources/training_logs"), exp_name)
    ckpt_path = _get_directory(Path("resources/checkpoints"), exp_name)
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    if continue_ckpt is not None:
        # ensure exists
        ckpt_path_old = Path(continue_ckpt)
        assert ckpt_path_old.exists(), f"Checkpoint path {ckpt_path_old} does not exist."

    if flags_dictionary["pruned_ff_name"] is None :
        focus_dir = ckpt_path
    else:
        focus_dir = flags_dictionary["focus_dir"]

    yaml_path = Path(ckpt_path, f"{exp_name}_training_status.yaml")
    # if continue_ckpt is None:
    rgb_yaml = 'used' if flags_dictionary["rgb"] else 'not used'
    flags = {
        'game': flags_dictionary["game"],
        'seed': flags_dictionary["seed"],
        'environments': flags_dictionary["environments"],
        # 'reward': flags_dictionary["reward"], #TODO: when reward and reward_mode?
        'reward': flags_dictionary["reward"], #TODO: when reward and reward_mode?
        'prune': flags_dictionary["pruned_ff_name"],
        'exclude_properties': flags_dictionary["hide_properties"],
        'rgb': rgb_yaml,
        'normalize': flags_dictionary["normalize"],
        'hud': flags_dictionary["hud"],
        'continue_ckpt': flags_dictionary["continue_ckpt"],
    }
    _create_yaml(flags, yaml_path)




    def make_env(rank: int = 0, seed: int = 0, silent=False, refresh=True) -> Callable:
        def _init() -> gym.Env:
            env = Environment(flags_dictionary["env"],
                              seed=seed + rank,
                              focus_dir=focus_dir,
                              focus_file=flags_dictionary["pruned_ff_name"],
                              hide_properties=flags_dictionary["hide_properties"],
                              silent=silent,
                              refresh_yaml=refresh,
                              reward=flags_dictionary["reward"],
                              normalize=flags_dictionary["normalize"],
                              hud=flags_dictionary["hud"])
            env = EpisodicLifeEnv(env=env)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init

    def make_eval_env(rank: int = 0, seed: int = 0, silent=False, refresh=True) -> Callable:
        def _init() -> gym.Env:
            env = Environment(flags_dictionary["env"],
                              seed=seed + rank,
                              focus_dir=focus_dir,
                              focus_file=flags_dictionary["pruned_ff_name"],
                              hide_properties=flags_dictionary["hide_properties"],
                              silent=silent,
                              reward=0, #always env reward for eval
                              refresh_yaml=refresh,
                              normalize=flags_dictionary["normalize"],
                              hud=flags_dictionary["hud"])
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # preprocessing based on atari wrapper of the openai baseline implementation (https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py)
    if flags_dictionary["rgb"]:
        # NoopResetEnv not required, because v5 has sticky actions, and also frame_skip=5, so also not required to set in wrapper (1 means no frameskip). no reward clipping, because scobots dont clip as well
        train_wrapper_params = {"noop_max" : 0, "frame_skip" : 1, "screen_size": 84, "terminal_on_life_loss": True, "clip_reward" : False} # remaining values are part of AtariWrapper
        train_env = make_vec_env(flags_dictionary["env"], n_envs=n_envs, seed=int(flags_dictionary["seed"]),  wrapper_class=AtariWrapper, wrapper_kwargs=train_wrapper_params, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        train_env = VecTransposeImage(train_env) #required for PyTorch convolution layers.
        # disable EpisodicLifeEnv, ClipRewardEnv for evaluation
        eval_wrapper_params = {"noop_max" : 0, "frame_skip" : 1, "screen_size": 84, "terminal_on_life_loss": False, "clip_reward" : False} # remaining values are part of AtariWrapper
        eval_env = make_vec_env(flags_dictionary["env"], n_envs=n_eval_envs, seed=eval_env_seed, wrapper_class=AtariWrapper, wrapper_kwargs=eval_wrapper_params, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        eval_env = VecTransposeImage(eval_env) #required for PyTorch convolution layers.
    else:
        # check if compatible gym env
        monitor = make_env()()
        check_env(monitor.env)
        del monitor
        # silent init and dont refresh default yaml file because it causes spam and issues with multiprocessing
        # import pdb; pdb.set_trace()
        # eval_env = VecNormalize(SubprocVecEnv([make_eval_env(rank=i, seed=eval_env_seed, silent=True, refresh=False) for i in range(n_eval_envs)], start_method=MULTIPROCESSING_START_METHOD), norm_reward=False, training=False)
        # train_env = VecNormalize(SubprocVecEnv([make_env(rank=i, seed=int(flags_dictionary["seed"]), silent=True, refresh=False) for i in range(n_envs)], start_method=MULTIPROCESSING_START_METHOD), norm_reward=False)
        print(f"starting {n_eval_envs} evaluators and {n_envs} actors")
        eval_env = VecNormalize(SubprocVecEnv([make_eval_env(rank=i, seed=eval_env_seed, silent=True, refresh=False) for i in range(n_eval_envs)]), norm_reward=False, training=False)
        train_env = VecNormalize(SubprocVecEnv([make_env(rank=i, seed=int(flags_dictionary["seed"]), silent=True, refresh=False) for i in range(n_envs)]), norm_reward=False)

    rtpt_iters = training_timestamps // rtpt_frequency
    save_bm = SaveBestModelCallback(ckpt_path, rgb=flags_dictionary["rgb_exp"])
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

    tb_callback = TensorboardCallback(n_envs=n_envs)
    cbl = [checkpoint_callback, eval_callback, n_callback, tb_callback]
    if flags_dictionary["rgb_exp"]: #remove tb callback if rgb
        cbl = cbl[:-1]
    cb_list = CallbackList(cbl)
    new_logger = configure(str(log_path), ["tensorboard"])

    if continue_ckpt is not None:
        if flags_dictionary["rgb"]:
            raise NotImplementedError("RGB training continuation not implemented.")

        # find highest checkpoint in training_checkpoints
        all_files = list(ckpt_path_old.glob("training_checkpoints/*"))
        ckpt_files = [f for f in all_files if f.is_file() and f.suffix == ".zip"]
        vecnorm_files = [f for f in all_files if f.is_file() and f.suffix == ".pkl"]
        ckpt_files.sort(key=lambda x: int(x.stem.split("_")[-2]))
        vecnorm_files.sort(key=lambda x: int(x.stem.split("_")[-2]))
        if len(ckpt_files) > 0 and len(vecnorm_files) > 0:
            train_env = VecNormalize.load(str(vecnorm_files[-1]), train_env.venv)
            model = PPO.load(str(ckpt_files[-1]), env=train_env)
        else:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_path_old / 'training_checkpoints'}")

    elif flags_dictionary["rgb_exp"]:
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
        if flags_dictionary["game"] in ["Bowling", "Tennis"]:
            adam_step_size = 0.00025
            print(f"Using lower learning rate for {flags_dictionary['game']}.")
        if flags_dictionary["game"] in ["Kangaroo"]:
            adam_step_size = 0.0025
            print(f"Using higher learning rate for {flags_dictionary['game']}.")
        clipping_eps = 0.1
        model = PPO(
            policy_str,
            n_steps=2048,
            # n_steps=4096,
            # learning_rate=linear_schedule(adam_step_size),
            learning_rate=get_exponential_schedule(adam_step_size, 0.25),
            n_epochs=3,
            # batch_size=512,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(clipping_eps),
            vf_coef=1,
            normalize_advantage=True, 
            # vf_coef=0.5,
            ent_coef=0.01,
            # ent_coef=0.05,
            env=train_env,
            policy_kwargs=pkwargs,
            verbose=1)
        
    model.set_logger(new_logger)
    print(model.policy)
    print(f"Experiment name: {exp_name}")
    print(f"Started {type(model).__name__} training with {n_envs} actors and {n_eval_envs} evaluators...")

    if flags_dictionary["pruned_ff_name"] is not None:
        focus_file_path = Path(flags_dictionary["focus_dir"]) / flags_dictionary["pruned_ff_name"]
        shutil.copy(focus_file_path, ckpt_path / focus_file_path.name)
    model.learn(total_timesteps=training_timestamps, callback=cb_list, progress_bar=flags_dictionary["progress"])

    _update_yaml(yaml_path, model.num_timesteps, True)

if __name__ == '__main__':
    main()
