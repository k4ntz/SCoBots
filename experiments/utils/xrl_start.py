import torch
import matplotlib.pyplot as plt
import numpy as np

from torchinfo import summary
from tqdm import tqdm
from rtpt import RTPT
from time import perf_counter, sleep

from algos import reinforce
from algos import genetic_rl as genetic
from scobi import Environment
from captum.attr import IntegratedGradients
from pathlib import Path
import matplotlib as mpl
import json
import os
mpl.use("TkAgg")
mpl.rcParams['toolbar'] = 'None' 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def re_select_action(*args, **kwargs):
    return reinforce.select_action(*args, **kwargs)


pause = False
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    global pause
    if event.button == 1:
        pause = not pause
    elif pause and event.button == 3:
        i = 0
        while True:
            savename = f"explanations_{i}.svg"
            if not os.path.exists(savename):
                break
            i += 1
        plt.savefig(savename)
        print(f"Saving current image in {savename}")


# function to test agent loaded via main switch
def play_agent(cfg, model, select_action_func, normalizer, epochs):
    runs = 3
    # init env
    draw = cfg.liveplot
    env = Environment(cfg.env_name,
                      cfg.seed,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file,
                      draw_features=draw)
    n_actions = env.action_space.n
    _, ep_reward = env.reset(), 0
    obs, _, _, _, _, info, obs_raw = env.step(1)
    features = obs
    summary(model, input_size=(1, len(features)), device=cfg.device)
    print("Runs:", runs)
    rewards = []
    all_sco_rewards = []
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name + "_EVAL", max_iterations=runs)
    rtpt.start()
    model.to(dev)
    ig = IntegratedGradients(model)
    fps = 30
    frame_delta = 1.0 / fps
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    zoom = 8
    fsize = (obs_raw.shape[1]+0) * px * zoom, (obs_raw.shape[0]+ 0) * px * zoom
    fig, axes = plt.subplots(2, 2, figsize=fsize, height_ratios=[1, 5])
    for ax in axes.flatten():
        ax.axis("off")
    rows, cells, colors = [], [], []
    columns = ["X, Y", "W, H", "R, G, B"]
    for obj in range(9):
        rows.append("category")
        cells.append(["xy", "wh", "rgb"])
        colors.append([1, 1, 1, 1])
    my_table = axes[0][0].table(cellText=cells,
                                rowLabels=rows,
                                rowColours=colors,
                                colLabels=columns,
                                colWidths=[.2,.2,.3],
                                cellLoc ='center',
                                loc='center')
    my_table.set_fontsize(80)
    my_table.scale(1, 2.3)
    img = axes[1][0].imshow(obs_raw, interpolation='none')
    features_text = axes[0][1].text(.2, 0., '', fontsize=22)
    img2 = axes[1][1].imshow(obs_raw, interpolation='none')
    # plt.tight_layout()
    plt.ion()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    max_nb_row = 0
    outfile = "obs.npy"
    out_array = []
    for run in tqdm(range(runs)):
        # env loop
        t = 0
        ep_reward = 0
        sco_reward = 0
        env.reset()
        while t < cfg.train.max_steps_per_trajectory:  # Don't infinite loop while playing
            if not pause:
                features = normalizer.normalize(features)
                out_array.append(features)
                action, _, probs = select_action_func(features, model, 0.05, n_actions)
                input = torch.tensor(features, requires_grad=True).unsqueeze(0).to(dev)
                output = int(np.argmax(probs[0]))
                attris = ig.attribute(input, target=output, method="gausslegendre")
                if cfg.liveplot:
                    img.set_data(env._obj_obs)
                    img2.set_data(env._rel_obs)
                    # table
                    nb_row = len(my_table.get_celld()) // 4
                    if max_nb_row < nb_row:
                        max_nb_row = nb_row
                        plt.tight_layout()
                    for i, obj in enumerate(env.oc_env.objects):
                        if i+1 > nb_row:
                            height = my_table.get_celld()[(1, -1)].get_height()
                            my_table.add_cell(i+1, -1, width=0.2, height=height, text=obj.category, loc="center", facecolor=to_rgba(obj.rgb))
                            my_table.add_cell(i+1, 0, width=0.2, height=height, text=obj.xy, loc="center")
                            my_table.add_cell(i+1, 1, width=0.2, height=height, text=obj.wh, loc="center")
                            my_table.add_cell(i+1, 2, width=0.3, height=height, text=obj.rgb, loc="center")
                        else:
                            my_table.get_celld()[(i+1, -1)].get_text().set_text(obj.category)
                            my_table.get_celld()[(i+1, -1)].set_color(to_rgba(obj.rgb))
                            my_table.get_celld()[(i+1, 0)].get_text().set_text(obj.xy)
                            my_table.get_celld()[(i+1, 1)].get_text().set_text(obj.wh)
                            my_table.get_celld()[(i+1, 2)].get_text().set_text(obj.rgb)
                    if len(env.oc_env.objects) > 1:
                        while nb_row > i+1:
                            del my_table._cells[(nb_row, -1)]
                            del my_table._cells[(nb_row, 0)]
                            del my_table._cells[(nb_row, 1)]
                            del my_table._cells[(nb_row, 2)]
                            nb_row -= 1
                    # rightings
                    to_draw = "\n".join(env._top_features) + "\n\n"
                    if action:
                        to_draw += f" --> {env.action_space_description[action]}"
                    features_text.set_text(to_draw)
                    plt.pause(frame_delta)
                env.set_feature_attribution(attris.squeeze(0).detach().cpu().numpy())
                obs, reward, scobi_reward, done, done2, info, obs_raw = env.step(action)
                #print(scobi_reward)
                features = obs
                ep_reward += reward
                sco_reward += scobi_reward
                t += 1
                if done or done2:
                    break
            fig.canvas.get_tk_widget().update()
        rewards.append(ep_reward)
        all_sco_rewards.append(sco_reward)
        np.save(outfile, out_array)
        rtpt.step()
    print(rewards)
    print(all_sco_rewards)
    print("Mean of Env Rewards:", sum(rewards) / len(rewards))
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

# function to call reinforce algorithm
def use_reinforce(cfg, mode):
    if mode == "train":
        if "Kangaroo" in cfg.env_name:
            reinforce.train_kangaroo(cfg)
        else:
            reinforce.train(cfg)
    elif mode == "eval":
        model, normalizer, epochs = reinforce.eval_load(cfg)
        model.eval()
        play_agent(cfg, model, re_select_action, normalizer, epochs)
    elif mode == "discover":
        reinforce.eval_reward_discovery(cfg)
    else:
        pass
        # explain(agent=agent, cfg=cfg)


# function to call deep neuroevolution algorithm
def use_genetic(cfg, mode):
    print("Selected algorithm: Deep Neuroevolution")
    if mode == "train":
        #TODO: silent mode not working for genetic only
        genetic.train(cfg)
    else:
        model, gen_select_action = genetic.eval_load(cfg)
        if mode == "eval":
            play_agent(cfg, model, gen_select_action)
        elif mode == "explain":
            pass
            # explain(agent=agent, cfg=cfg)


def to_rgba(color):
    return np.concatenate([np.array(color)/255, [.5]])

# main function
# switch for each algo
def xrl(cfg, mode):
    # algo selection
    # 1: REINFORCE
    # 2: Deep Neuroevolution
    if cfg.rl_algo == 1:
        use_reinforce(cfg, mode)
    elif cfg.rl_algo == 2:
        use_genetic(cfg, mode)
    else:
        print("Unknown algorithm selected")