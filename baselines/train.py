import argparse
import gymnasium as gym
import numpy as np
import os
from scobi import Environment
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, WarpFrame, AtariWrapper
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback, CallbackList, EvalCallback
from pathlib import Path
from typing import Callable
from rtpt import RTPT
from collections import deque
import torch as th


MULTIPROCESSING_START_METHOD = "spawn" # if os.name == 'nt' else "fork"  # 'nt' == Windows

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
        return True

    
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
        return True
        


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func



def get_settings_str(opts):
    settings_str = ""
    if opts.reward == "env":
        settings_str += "_re"
    if opts.reward == "human":
        settings_str += "_rh"
    if opts.reward == "mixed":
        settings_str += "_rm"

    if opts.prune == "no_prune":
        settings_str += "_pr-nop"
    if opts.prune == "default":
        settings_str += "_pr-def"
    if opts.prune == "external":
        settings_str += "_pr-ext"

    if opts.exclude_properties:
        settings_str += '_ep'

    if opts.input_data == "SPACE":
        settings_str += "_SPACEinput"
    if opts.input_data == "OCAtari":
        settings_str += "_OCAtariinput"
    
    if opts.num_layers == 1:
        settings_str += "_1l"
    if opts.num_layers == 2:
        settings_str += "_2l"


    if opts.rgbv5:
        settings_str = "-rgb-v5"
    elif opts.rgbv4:
        settings_str = "-rgb-v4"
    else:
        settings_str += "-v3"

    return settings_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-c", "--cores", type=int, required=True,
                        help="number of envs used")
    parser.add_argument("-r", "--reward", type=str, default="env", choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, default= "no_prune", choices=["no_prune", "default", "external"], 
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'baselines_focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-e", "--exclude_properties", action="store_true", help="exclude properties from feature vector")
    parser.add_argument("--rgbv4", action="store_true", help="rgb observation space")
    parser.add_argument("--rgbv5", action="store_true", help="rgb observation space")
    parser.add_argument("--use_checkpoint", action="store_true", help="use checkpoint")
    parser.add_argument("--num_layers", type=int, choices=[1, 2], default=2, help="number of layers for mlp policy")
    parser.add_argument("--adam_step_size", type=float, choices=[0.00025, 0.001], default=0.00025, help="adam step size")
    parser.add_argument("--input_data", type=str, choices=["SPACE", "OCAtari",], default="SPACE", help="input data")
    opts = parser.parse_args()


    settings_str = get_settings_str(opts)
    print(settings_str)
    
    env_str = "ALE/" + opts.game +"-v5"
    if opts.rgbv4 and opts.rgbv5:
        print("please select only one rgb mode!")
    #override some settings if rgb
    rgb_exp = opts.rgbv4 or opts.rgbv5
    if opts.rgbv4:
        env_str = opts.game + "NoFrameskip-v4"
    if opts.rgbv5:
        pass
    
    reward_mode = 0
    if opts.reward == "env":
        pass
    if opts.reward == "human":
        reward_mode = 1
    if opts.reward == "mixed":
        reward_mode = 2

    
    pruned_ff_name = None
    game_id = env_str.split("/")[-1].lower().split("-")[0]
    if opts.prune in ["default", "external"]:
        pruned_ff_name = f"pruned_{game_id}.yaml"

    focus_dir = "focusfiles"
    if opts.prune == "default":
        focus_dir = "focusfiles"
    if opts.prune == "external":
        focus_dir = "baselines_focusfiles"


    hide_properties = False
    if opts.exclude_properties:
        hide_properties = True


    n_envs = opts.cores
    n_eval_envs = 4 #4
    n_eval_episodes = 8
    eval_env_seed = (opts.seed + 42) * 2 #different seeds for eval
    training_timestamps = 10_000_000 #20_000_000 # was 20_000_000
    checkpoint_frequency = 200_000 #1_000_000
    eval_frequency = 400_000 #500_000
    rtpt_frequency = 100_000 # 100_000

    exp_name = opts.game + "_s" + str(opts.seed) + settings_str



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
                              refresh_yaml=refresh,
                              object_detector=opts.input_data)
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
                              reward=0, #always env reward for eval
                              refresh_yaml=refresh,
                              object_detector=opts.input_data)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # preprocessing based on atari wrapper of the openai baseline implementation (https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py)
    if opts.rgbv4:
        # NoopResetEnv:30, MaxAndSkipEnv=4, WarpFrame=84x84,grayscale, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv, frame_stack=1, scale=False
        train_wrapper_params = {"noop_max" : 30, "frame_skip" : 4, "screen_size": 84, "terminal_on_life_loss": True, "clip_reward" : True, "action_repeat_probability" : 0.0} # remaining values are part of AtariWrapper
        train_env = make_vec_env(env_str, n_envs=n_envs, seed=opts.seed,  wrapper_class=AtariWrapper, wrapper_kwargs=train_wrapper_params, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        train_env = VecTransposeImage(train_env) #required for PyTorch convolution layers.
        # disable EpisodicLifeEnv, ClipRewardEnv for evaluation
        eval_wrapper_params = {"noop_max" : 30, "frame_skip" : 4, "screen_size": 84, "terminal_on_life_loss": False, "clip_reward" : False, "action_repeat_probability" : 0.0} # remaining values are part of AtariWrapper
        eval_env = make_vec_env(env_str, n_envs=n_eval_envs, seed=eval_env_seed, wrapper_class=AtariWrapper, wrapper_kwargs=eval_wrapper_params, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
        eval_env = VecTransposeImage(eval_env) #required for PyTorch convolution layers.
    elif opts.rgbv5:
        # NoopResetEnv not required, because v5 has sticky actions, and also frame_skip=5, so also not required to set in wrapper (1 means no frameskip). no reward clipping, because scobots dont clip as well
        train_wrapper_params = {"noop_max" : 0, "frame_skip" : 1, "screen_size": 84, "terminal_on_life_loss": True, "clip_reward" : False} # remaining values are part of AtariWrapper
        train_env = make_vec_env(env_str, n_envs=n_envs, seed=opts.seed,  wrapper_class=AtariWrapper, wrapper_kwargs=train_wrapper_params, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method" :"fork"})
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
        train_env = VecNormalize(SubprocVecEnv([make_env(rank=i, seed=opts.seed, silent=True, refresh=False) for i in range(n_envs)], start_method=MULTIPROCESSING_START_METHOD), norm_reward=False)

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
        if opts.rgbv4:
            norm_adv = False # according to sb3, adv normalization not used in original ppo paper
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
        if opts.num_layers == 2:
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        elif opts.num_layers == 1:
            net_arch = dict(pi=[64], vf=[64])
        else:
            raise ValueError("num_layers must be 1 or 2")
        activation_fn = th.nn.ReLU
        policy_str = "MlpPolicy"
        pkwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
        adam_step_size = opts.adam_step_size
        clipping_eps = 0.1

        if not opts.use_checkpoint:
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
        else:
            # TODO check whether hyperparameters are loaded correctly
            
            # load last checkpoint
            ckpt_list = [f for f in os.listdir(ckpt_path) if f.endswith(".zip") and f.startswith("model")]
            last_ckpt = sorted(ckpt_list, key=lambda x: int(x.split("_")[1]))[-1]
            #remove .zip
            last_ckpt = last_ckpt[:-4]
            vec_norm_list = [f for f in os.listdir(ckpt_path) if f.endswith(".pkl") and f.startswith("model_vecnormalize")]
            last_vecnorm = sorted(vec_norm_list, key=lambda x: int(x.split("_")[2]))[-1]
            eval_env = SubprocVecEnv([make_eval_env(rank=i, seed=eval_env_seed, silent=True, refresh=False) for i in range(n_eval_envs)], start_method=MULTIPROCESSING_START_METHOD)
            train_env = SubprocVecEnv([make_env(rank=i, seed=opts.seed, silent=True, refresh=False) for i in range(n_envs)], start_method=MULTIPROCESSING_START_METHOD)
            train_env = VecNormalize.load(Path(ckpt_path, last_vecnorm), train_env)
            eval_env = VecNormalize.load(Path(ckpt_path, last_vecnorm), eval_env)
            train_env.training = True
            eval_env.training = False
            train_env.norm_reward = False
            eval_env.norm_reward = False
            print(f"loading model from {last_ckpt}")
            model_path = Path(ckpt_path, last_ckpt)

            model = PPO.load(model_path,
                             env=train_env,)
            #                  clip_range=linear_schedule((steps_left/training_timestamps) * clipping_eps),
            #                  adam_step_size = linear_schedule((steps_left/training_timestamps) * adam_step_size),
            # )
            steps_left = training_timestamps - int(last_ckpt.split("_")[1])
            training_timestamps = steps_left
            
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
    if rgb_exp: #remove tb callback if rgb
        cbl = cbl[:-1]
    cb_list = CallbackList(cbl)


    model.set_logger(new_logger)
    print(model.policy)
    print(f"Experiment name: {exp_name}")
    print(f"Started {type(model).__name__} training with {n_envs} actors and {n_eval_envs} evaluators...")

    reset_num_timesteps = False
    if opts.use_checkpoint:
        print("Continuing training from last checkpoint")
        reset_num_timesteps = True

    model.learn(total_timesteps=training_timestamps, callback=cb_list, progress_bar=False, reset_num_timesteps=reset_num_timesteps)

if __name__ == '__main__':
    main()