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

def load_ppo_env(focus_dir = "../baselines_focusfiles", ff_name = "pruned_pong.yaml", exp_name = "Pong_s42_re-v3_gtdata", ckpt_path = "../baselines_checkpoints",
                 as_vecenv=False):
    env_str = "Pong-v4"
    #ff_name = "default_focus_Pong-v4.yaml"
    #focus_dir = "../focusfiles"
    hide_properties = False
    env = Environment(env_str,
                        focus_dir=focus_dir,
                        focus_file=ff_name, 
                        hide_properties=hide_properties, 
                        draw_features=True, # implement feature attribution
                        reward=0) #env reward only for evaluation
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

def ppo_select_action(features, policy, random_tr = -1, n_actions=3):
    mlp_extractor = policy.mlp_extractor
    action_net = policy.action_net
    feature_tensor = torch.tensor(features, device=dev, dtype=torch.float32)
    mlp_extractor.eval()
    action_net.eval()
    with torch.no_grad():
        features = mlp_extractor.forward_actor(feature_tensor)
        logits = action_net(features)
        action = torch.argmax(logits)
        return action.item(), None, None
    
def ppo_select_action_v2(features, policy, random_tr = -1, n_actions=3):
    policy.eval()
    with torch.no_grad():
        features = torch.tensor(features, device=dev, dtype=torch.float32).unsqueeze(0)
        actions, values, log_prob = policy.forward(features, deterministic=True)
        return actions.item(), values.item(), log_prob.item()

def ppo_load_model_v2(cfg):
    focus_filename = "pruned_pong.yaml"
    focus_dir = "../baselines_focusfiles"
    input_size, output_size = get_input_and_output_size(focus_filename, focus_dir)
    model_folder_path = "../ppo_pruned_inx64xout/"
    net_arch=dict(pi=[64], vf=[64])
    return ppp_load_weights(model_folder_path, net_arch, input_size, output_size)

def ppo_load_model(cfg):
    focus_filename = "default_focus_Pong-v4.yaml"
    focus_dir = "../focusfiles"
    input_size, output_size = get_input_and_output_size(focus_filename, focus_dir)
    model_folder_path = "../"
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    return ppp_load_weights(model_folder_path, net_arch, input_size, output_size)

def get_input_and_output_size(focus_filename, focus_dir):
    focus_object = Focus(fofiles_dir_name=focus_dir, fofile=focus_filename)
    actions = focus_object.PARSED_ACTIONS
    fnames = focus_object.get_vector_entry_descriptions()
    fnames = [f.replace(" ", "") for f in fnames]
    input_size = len(fnames)
    output_size = len(actions)
    return input_size, output_size

def ppp_load_weights(model_folder_path, net_arch, input_size, output_size):
    checkpoint = torch.load(model_folder_path + "model.pth") # was "checkpoint.pth"
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