import torch
import matplotlib.pyplot as plt
import numpy as np

try:
    from torchinfo import summary
    from captum.attr import IntegratedGradients
except:
    pass
from tqdm import tqdm
from rtpt import RTPT


from scobi import Environment

from pathlib import Path
import matplotlib as mpl
import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from experiments.utils.xrl_utils import Drawer

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def play_agent(cfg, model, select_action_func, normalizer, epochs, env=None, collect_best_policy_states_and_actions=False):
    INTEGRATED_GRADIENTS = False
    runs = 150
    # init env
    draw = cfg.liveplot
    if env is None:
        env = Environment(cfg.env_name,
                          cfg.seed,
                          reward=cfg.scobi_reward_shaping,
                          hide_properties=cfg.scobi_hide_properties,
                          focus_dir=cfg.scobi_focus_dir,
                          focus_file=cfg.scobi_focus_file,
                          draw_features=draw)
    _, ep_reward = env.reset(), 0
    obs, _, _, _, info = env.step(1)
    obs_raw = env.original_obs
    features = obs
    #summary(model, input_size=(1, len(features)), device=cfg.device)
    print("Runs:", runs)
    rewards = []
    all_sco_rewards = []
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name + "_EVAL", max_iterations=runs)
    rtpt.start()
    model.to(dev)
    if INTEGRATED_GRADIENTS:
        ig = IntegratedGradients(model)
    
    # initialize drawing
    drawer = Drawer(obs_raw)

    # initialize output file
    if "eclaire" in cfg:
        print("eclaire_dir: ", cfg.eclaire.eclaire_dir)
        outfile_path = os.path.join("..", cfg.eclaire.eclaire_dir, "obs.npy")
        if collect_best_policy_states_and_actions:
            outfile_path = os.path.join("..", cfg.eclaire.eclaire_dir, "best_policy_obs.npy")

    else:
        outfile_path = "obs.npy"
        Path(outfile_path).unlink(missing_ok=True)
    out_array = [] # TODO: check correctness: moved to here from inside first loop
    actions = []

    for run in tqdm(range(runs)):
        if "eclaire" in cfg:
            if len(out_array) >= cfg.eclaire.num_samples:
                break
        t = 0
        ep_reward = 0
        sco_reward = 0
        env.reset(seed=cfg.seed + run)
        while t < cfg.train.max_steps_per_trajectory:  # Don't infinite loop while playing
            if not drawer.pause:
                features = normalizer.normalize(features)
                out_array.append(features) # save normalized features
                if not collect_best_policy_states_and_actions:
                    action, _, probs = select_action_func(features, model, 0.25, n_actions = env.action_space.n) # select random action with 25% probability to create diversity in data
                else:   
                    action, _, probs = select_action_func(features, model, 0, n_actions = env.action_space.n)
                    actions.append(action)
                if INTEGRATED_GRADIENTS:
                    input = torch.tensor(features, requires_grad=True).unsqueeze(0).to(dev)
                    output = int(np.argmax(probs[0]))
                    attris = ig.attribute(input, target=output, method="gausslegendre")
                    env.set_feature_attribution(attris.squeeze(0).detach().cpu().numpy())
                
                if draw:
                    drawer.draw_play(action, env)

                obs, scobi_reward, done, done2, info = env.step(action)
                obs_raw = env.original_obs
                features = obs
                ep_reward += env.original_reward
                sco_reward += scobi_reward
                t += 1
                if done or done2:
                    break
            if draw:
                drawer.update(run, t)
        rewards.append(ep_reward)
        all_sco_rewards.append(sco_reward)
        rtpt.step()
    print(rewards)
    print(all_sco_rewards)
    print("Mean of Env Rewards:", sum(rewards) / len(rewards))
    
    #write results to file
    np.save(outfile_path, out_array)
    print("len(out_array): ", len(out_array))
    if collect_best_policy_states_and_actions:
        np.save(outfile_path.replace("best_policy_obs", "best_policy_actions"), actions)
        print("len(actions): ", len(actions))
    #write_results(cfg, runs, rewards, all_sco_rewards, epochs, out_array, outfile_path)


def write_results(cfg, runs, rewards, all_sco_rewards, epochs, out_array, outfile_path):
    result_file = Path(__file__).parent.parent / Path("results", "results.json")
    if result_file.exists():
       result_dict = json.load(open(result_file))
    else:
       result_dict = {cfg.env_name : {}}
       result_file.touch()

    mode_str = "base"
    if cfg.scobi_interactive:
       mode_str = "pruned"
    if cfg.scobi_reward_shaping:
       mode_str += "_reward"
    entry =  {
       "mode" : mode_str,
       "seed" : cfg.seed,
       "eval_runs" : runs,
       "reward": sum(rewards) / len(rewards),
       "scobi_reward": sum(all_sco_rewards) / len(all_sco_rewards),
       "frames_seen": epochs * cfg.train.steps_per_episode
       }
    if cfg.env_name not in result_dict.keys():
       result_dict[cfg.env_name] = {}
    result_dict[cfg.env_name][cfg.exp_name] = entry
    result_file.write_text(json.dumps(result_dict, indent=4))