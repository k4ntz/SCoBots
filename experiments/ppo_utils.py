try:
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
except:
    pass
from scobi import Environment
import gymnasium as gym
import torch
import numpy as np
from scobi.focus import Focus
import os


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_ppo_env(cfg, ckpt_path = "../baselines_checkpoints",
                 as_vecenv=False):
    focus_dir = os.path.join("..", cfg.eclaire.focus_dir)
    ff_name = cfg.eclaire.focus_filename
    exp_name = cfg.eclaire.eclaire_dir.split("_", 1)[1]
    env_str = exp_name.split("_", 1)[0] + "-v5"
    env_str = "ALE/" + env_str
    hide_properties = False
    # env_kwargs = {}
    # env_kwargs = {"frameskip": (5,6),
    #               "repeat_action_probability": 0.25}
    env = Environment(env_str,
                        focus_dir=focus_dir,
                        focus_file=ff_name, 
                        hide_properties=hide_properties, 
                        draw_features=True, # implement feature attribution
                        reward=0, #env reward only for evaluation
                        object_detector=cfg.eclaire.input_data,
                        )
    if as_vecenv:
        dummy_vecenv = DummyVecEnv([lambda :  env])
        vecnorm_path = os.path.join(ckpt_path, exp_name, "best_vecnormalize.pkl")
        EVAL_ENV_SEED = 42
        _, _ = env.reset(seed=EVAL_ENV_SEED)
        dummy_vecenv = DummyVecEnv([lambda :  env])
        env = VecNormalize.load(vecnorm_path, dummy_vecenv)
        env.training = False
        env.norm_reward = False
    return env

def ppo_select_action(features, policy, random_tr = 0, n_actions=3):
    mlp_extractor = policy.mlp_extractor
    action_net = policy.action_net
    feature_tensor = torch.tensor(features, device=dev, dtype=torch.float32)
    mlp_extractor.eval()
    action_net.eval()
    with torch.no_grad():
        features = mlp_extractor.forward_actor(feature_tensor)
        logits = action_net(features)
        action = torch.argmax(logits)
    if random_tr > 0:
        if np.random.rand() < random_tr:
            return np.random.randint(n_actions), None, None

    return action.item(), None, None
    
def ppo_select_action_v2(features, policy, random_tr = -1, n_actions=3):
    policy.eval()
    with torch.no_grad():
        features = torch.tensor(features, device=dev, dtype=torch.float32).unsqueeze(0)
        actions, values, log_prob = policy.forward(features, deterministic=True)
        return actions.item(), values.item(), log_prob.item()

def ppo_load_model(cfg):
    focus_filename = cfg.eclaire.focus_filename
    focus_dir = os.path.join("..", cfg.eclaire.focus_dir)
    input_size, output_size = get_input_and_output_size(focus_filename, focus_dir)
    model_folder_path = os.path.join("..", cfg.eclaire.eclaire_dir, cfg.eclaire.model_filename)
    net_arch = dict(pi=[64]*cfg.eclaire.num_layers, vf=[64]*cfg.eclaire.num_layers)
    return ppo_load_weights(model_folder_path, net_arch, input_size, output_size)

def get_input_and_output_size(focus_filename, focus_dir):
    focus_object = Focus(fofiles_dir_name=focus_dir, fofile=focus_filename)
    actions = focus_object.PARSED_ACTIONS
    fnames = focus_object.get_vector_entry_descriptions()
    fnames = [f.replace(" ", "") for f in fnames]
    input_size = len(fnames)
    output_size = len(actions)
    return input_size, output_size

def ppo_load_weights(model_file_path, net_arch, input_size, output_size):
    checkpoint = torch.load(model_file_path) # was "checkpoint.pth"
    pkwargs = dict(activation_fn=torch.nn.ReLU, net_arch=net_arch)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(input_size,), dtype=np.float32)
    action_space = gym.spaces.Discrete(output_size)
    policy_net = ActorCriticPolicy(
        observation_space,
        action_space,
        BasePolicy._dummy_schedule,
        net_arch= pkwargs["net_arch"],
        activation_fn=pkwargs["activation_fn"],
    )
    policy_net.load_state_dict(checkpoint)
    return policy_net