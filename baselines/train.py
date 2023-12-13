import argparse
import gymnasium as gym
import numpy as np
import os
from scobi import Environment
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback, CallbackList, EvalCallback
from pathlib import Path
from typing import Callable
from rtpt import RTPT
from collections import deque
import torch as th


MULTIPROCESSING_START_METHOD = "spawn" if os.name == 'nt' else "fork"  # 'nt' == Windows


class rgbWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env, screen_size: int = 84) -> None:
        env = EpisodicLifeEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        super().__init__(env)


class RtptCallback(BaseCallback):
    def __init__(self, exp_name, max_iter, verbose=0):
        super(RtptCallback, self).__init__(verbose)
        self.rtpt = RTPT(name_initials="QD",
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
        


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-c", "--cores", type=int, required=True,
                        help="number of envs used")
    parser.add_argument("-r", "--reward", type=str, required=False, choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, required=False, choices=["default", "external"], 
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'baselines_focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-e", "--exclude_properties", action="store_true", help="exclude properties from feature vector")
    parser.add_argument("--rgb", action="store_true", help="rgb observation space")
    opts = parser.parse_args()

    env_str = "ALE/" + opts.game +"-v5"
    settings_str = ""
    pruned_ff_name = None
    focus_dir = "focusfiles"
    hide_properties = False
    
    reward_mode = 0
    if opts.reward == "env":
        settings_str += "_re"
    if opts.reward == "human":
        settings_str += "_rh"
        reward_mode = 1
    if opts.reward == "mixed":
        settings_str += "_rm"
        reward_mode = 2

    game_id = env_str.split("/")[-1].lower().split("-")[0]

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
    if opts.prune == "default":
        settings_str += "_pr-def"
    if opts.prune == "external":
        settings_str += "_pr-ext"
        focus_dir = "baselines_focusfiles"
    if opts.exclude_properties:
        settings_str += '_ep'
        hide_properties = True


    n_envs = opts.cores
    n_eval_envs = 4
    n_eval_episodes = 8
    eval_env_seed = (opts.seed + 42) * 2 #different seeds for eval
    training_timestamps = 20_000_000
    checkpoint_frequency = 1_000_000
    eval_frequency = 500_000
    rtpt_frequency = 100_000

    #override some settings if rgb
    if opts.rgb:
        settings_str = "-rgb"
        training_timestamps = 10_000_000
    
    exp_name = opts.game + "_s" + str(opts.seed) + settings_str + "-v3"
    log_path = Path("baselines_logs", exp_name)
    ckpt_path = Path("baselines_checkpoints", exp_name)
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    def make_env(rank: int = 0, seed: int = 0, silent=False, refresh=True) -> Callable:
        def _init() -> gym.Env:
            env = Environment(env_str, 
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
                              focus_dir=focus_dir,
                              focus_file=pruned_ff_name, 
                              hide_properties=hide_properties, 
                              silent=silent,
                              reward=0,
                              refresh_yaml=refresh) #always env reward for eval
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init


    if opts.rgb:
        # atari wrap around v5: keep 84x84, graysacle, termination signal
        eval_env = make_vec_env(env_str, n_envs=n_eval_envs, seed=eval_env_seed, wrapper_class=rgbWrapper, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        eval_env = VecFrameStack(eval_env, n_stack=4)
        train_env = make_vec_env(env_str, n_envs=n_envs, seed=opts.seed, wrapper_class=rgbWrapper, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        train_env = VecFrameStack(train_env, n_stack=4)
    else:
        # check if compatible gym env
        monitor = make_env()()
        check_env(monitor.env)
        del monitor
        # silent init and dont refresh default yaml file because it causes spam and issues with multiprocessing 
        eval_env = VecNormalize(SubprocVecEnv([make_eval_env(rank=i, seed=eval_env_seed, silent=True, refresh=False) for i in range(n_eval_envs)], start_method=MULTIPROCESSING_START_METHOD), norm_reward=False, training=False)
        train_env = VecNormalize(SubprocVecEnv([make_env(rank=i, seed=opts.seed, silent=True, refresh=False) for i in range(n_envs)], start_method=MULTIPROCESSING_START_METHOD), norm_reward=False)

    rtpt_iters = training_timestamps // rtpt_frequency
    save_bm = SaveBestModelCallback(ckpt_path, rgb=opts.rgb)
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
        save_path=str(ckpt_path),
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
    if opts.rgb: #remove tb callback if rgb
        cbl = cbl[:-1]
    cb_list = CallbackList(cbl)
    new_logger = configure(str(log_path), ["tensorboard"])

    if opts.rgb:
        policy_str = "CnnPolicy"
        adam_step_size = 0.00025
        clipping_eps = 0.1
        model = PPO(
            policy=policy_str,
            n_steps=128,
            learning_rate=linear_schedule(adam_step_size),
            n_epochs=3,
            batch_size=32*8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(clipping_eps),
            vf_coef=1,
            ent_coef=0.01,
            env=train_env,
            verbose=1)
    else:
        policy_str = "MlpPolicy"
        pkwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[64, 64], vf=[64, 64]))
        adam_step_size = 0.00025 # or 0.001
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

if __name__ == '__main__':
    main()