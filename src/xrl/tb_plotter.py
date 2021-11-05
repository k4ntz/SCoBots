from packaging import version

import os
import pandas as pd
import numpy as np
import seaborn as sns
import tensorboard as tb

from matplotlib import pyplot as plt
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.signal import savgol_filter

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

sns.set_style("ticks")

# helper function to smooth plots
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# https://stackoverflow.com/questions/46633544/smoothing-out-curve-in-python
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
def smooth2(y, window=21, order=2):
    return savgol_filter(y, window, order)


# plots for genetic with raw features
def plot_gen_raw():
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    fig, axs = plt.subplots(ncols=2, figsize=(12,4))
    ### genetic with raw features
    run = "gen-pong-raw"
    x = EventAccumulator(path=os.getcwd() + "/xrl/genlogs/" + run + "/")
    x.Reload()
    # reward for genetic top5
    df_r = pd.DataFrame(x.Scalars('Train/Mean of top 5'))
    # smooth
    df_r["s-value"] = smooth2(df_r["value"], 121)
    p2 = sns.lineplot(data=df_r, x="step", y="value", alpha=0.2, legend=False, color="green", ax=axs[0])
    sns.lineplot(data=df_r, x="step", y="s-value", color="green", ax=axs[0]).set_title("Mean of top 5")
    p2.set(ylabel='Reward', xlabel='Generation')
    # reward for genetic generation
    df_r = pd.DataFrame(x.Scalars('Train/Mean rewards'))
    # smooth
    df_r["s-value"] = smooth2(df_r["value"], 121)
    p3= sns.lineplot(data=df_r, x="step", y="value", alpha=0.2, legend=False, color="green", ax=axs[1])
    sns.lineplot(data=df_r, x="step", y="s-value", color="green", ax=axs[1]).set_title("Mean of generation")
    p3.set(ylabel='Reward', xlabel='Generation')
    # finish plot
    sns.despine(offset=1, trim=True)
    plt.tight_layout()
    #plt.savefig("reward.pdf")
    plt.show()


# function to plot space exp
##### DONE #######
def plot_space_exp():
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    runs = ["DQ-Learning-Pong-v8-cnn", "DQ-Learning-Pong-v9-zw", "DQ-Learning-Pong-v11r"]
    df_list = []
    for i, run in enumerate(runs):
        x = EventAccumulator(path=os.getcwd() + "/dqn/logs/" + run + "/")
        x.Reload()
        # print(x.Tags())
        tdf = pd.DataFrame(x.Scalars('Train/reward_episode'))
        tag = "DuelDQN w. sorted SPACE output"
        # sert right experiment tag
        if i == 0:
            tag = "Baseline with raw images"
        elif i == 1:
            tag = "MLP w. sorted SPACE output"
        tdf["Experiment"] = tag
        tdf["s-value"] = smooth2(tdf["value"], 121)
        #tdf["s-value"] = smooth(tdf["value"], 60)
        df_list.append(tdf)
    df = pd.concat(df_list)
    print(df)
    p = sns.lineplot(data=df, x="step", y=df["value"], hue="Experiment", alpha=0.2, legend=False)
    sns.lineplot(data=df, x="step", y=df["s-value"], hue="Experiment", linewidth=2)
    p.set(ylabel='Reward', xlabel='Episode')
    sns.despine(offset=1, trim=True)
    plt.tight_layout()
    #plt.savefig("reward.pdf")
    plt.show()


# function to plot exp1
def plot_exp1():
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    fig, axs = plt.subplots(ncols=3, figsize=(15,5))
    ####### REINFORCE #######
    run = "re-pong-v2"
    x = EventAccumulator(path=os.getcwd() + "/xrl/relogs/" + run + "/")
    x.Reload()
    # reward for reinforce
    df_r = pd.DataFrame(x.Scalars('Train/Avg reward'))
    # smooth
    df_r["s-value"] = smooth2(df_r["value"], 121)
    # create both plots, one unsmoothed with alpha, one smoothed
    p = sns.lineplot(data=df_r, x="step", y="value", alpha=0.2, legend=False, color="red", ax=axs[0])
    sns.lineplot(data=df_r, x="step", y="s-value", color="red", ax=axs[0]).set_title("(A)")
    p.set(ylabel='Reward', xlabel='Episode')
    ####### GENETIC #######
    run = "gen-pong-v1"
    x = EventAccumulator(path=os.getcwd() + "/xrl/genlogs/" + run + "/")
    x.Reload()
    # reward for genetic top5
    df_r = pd.DataFrame(x.Scalars('Train/Mean of top 5'))
    # smooth
    df_r["s-value"] = smooth2(df_r["value"], 121)
    p2 = sns.lineplot(data=df_r, x="step", y="value", alpha=0.2, legend=False, color="green", ax=axs[1])
    sns.lineplot(data=df_r, x="step", y="s-value", color="green", ax=axs[1]).set_title("(B)")
    p2.set(ylabel='Reward', xlabel='Generation')
    # reward for genetic generation
    df_r = pd.DataFrame(x.Scalars('Train/Mean rewards'))
    # smooth
    df_r["s-value"] = smooth2(df_r["value"], 121)
    p3= sns.lineplot(data=df_r, x="step", y="value", alpha=0.2, legend=False, color="green", ax=axs[2])
    sns.lineplot(data=df_r, x="step", y="s-value", color="green", ax=axs[2]).set_title("(C)")
    p3.set(ylabel='Reward', xlabel='Generation')
    # finalize plots
    sns.despine(offset=1, trim=True)
    plt.tight_layout()
    #plt.savefig("dummy.pdf")
    plt.show()


# function to plot exp with first generalisation of meaningful features
# function to plot exp4
def plot_exp_general_f():
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    runs = ["exp2-re-pong-v2", "exp2-re-pong-v2-2", "exp2-re-pong-v2-3"]
    df_list = []
    for run in runs:
        x = EventAccumulator(path=os.getcwd() + "/xrl/relogs/" + run + "/")
        x.Reload()
        # print(x.Tags())
        tdf = pd.DataFrame(x.Scalars('Train/Avg reward'))
        df_list.append(tdf)
    df = pd.concat(df_list)
    sns.lineplot(data=df, x="step", y="value")
    # finalize plots
    sns.despine(offset=1, trim=True)
    plt.tight_layout()
    plt.show()



# function to plot exp4
def plot_exp4_tr():
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    runs = ["exp4-re-pong-ptr12", "exp4-re-pong-ptr12-3", "exp4-re-pong-ptr12-4"]
    df_list = []
    for i, run in enumerate(runs):
        x = EventAccumulator(path=os.getcwd() + "/xrl/relogs/" + run + "/")
        x.Reload()
        # print(x.Tags())
        tdf = pd.DataFrame(x.Scalars('Train/Avg reward'))
        tdf["Run"] = i
        df_list.append(tdf)
        tdf["s-value"] = smooth2(tdf["value"], 121)
    df = pd.concat(df_list)
    p = sns.lineplot(data=df, x="step", y="value", hue="Run", palette="deep", legend=False)
    #p = sns.lineplot(data=df, x="step", y="s-value", hue="Run", palette="deep", linewidth=2, legend=False)
    #p = sns.lineplot(data=df, x="step", y="value", linewidth=2, ci=2, palette="deep")
    p.set(ylabel='Reward', xlabel='Episode')
    # finish
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    #plt.savefig("exp4-trp-unsmoothed.pdf")
    plt.show()


# plotter for entropy
def plot_entropy():
    entropies = [2.564941380529404, 0.0, 4.67534368187212, 4.301236205046662, 4.786326586819943, 4.793996572835907, 4.751337825862695, 4.390388411732666, 4.297212849733501, 1.273596020172498, 4.301236205046662, 3.3669188324343584, 4.616563598171143, 4.63509707122358, 4.343800091706337, 4.361510531499313, 2.1129233462499477, 4.4921234695262715, 4.546718731753535, 4.402517607406379, 4.440621122918765]
    r_entropies = [2.7054705880877647, 0.0, 5.702558150850082, 3.7881597235512428, 5.659517017054085, 5.374671400816015, 5.103689456041231, 4.164941099731191, 4.189706288110571, 1.467541886419882, 3.7881597235512428, 3.0001041949036678, 5.10471241899767, 4.9509918604821195, 3.8009965553030645, 3.78320582949031, 1.6243743659617425, 4.523183311112938, 4.9854143456230045, 4.704306402337425, 4.5046693978854355]
    df = pd.DataFrame(r_entropies, index=[i for i in range(21)])
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    plt.figure(figsize=(6,5))
    # plot
    p = sns.histplot(r_entropies)
    p.set(xlabel='Entropy')
    # finish
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    #plt.savefig("dummy.pdf")
    plt.show()


# plot function for first try minidreamer
def plot_mdr():
    # first load all data
    runs = ["exp3-mdr-pong", "exp3-mdr-tennis"]
    list_dsi = []   # list distance next state
    list_dsns = []  # list distance id
    list_loss = []  # list loss 
    list_loss_s = []    # list loss state
    list_loss_r = []    # list loss reward
    for run in runs:
        x = EventAccumulator(path=os.getcwd() + "/xrl/mdrlogs/" + run + "/")
        x.Reload()
        # print(x.Tags())
        dsi_df = pd.DataFrame(x.Scalars('Train/Diff State Identity'))
        list_dsi.append(dsi_df)
        list_dsi[len(list_dsi) - 1]["run"] = run
        dsns_df = pd.DataFrame(x.Scalars('Train/Diff State Next State'))
        list_dsns.append(dsns_df)
        list_dsns[len(list_dsns) - 1]["run"] = run
        #list_loss.append(pd.DataFrame(x.Scalars('Train/Loss World Predictor')))
        #list_loss_s.append(pd.DataFrame(x.Scalars('Train/Loss World Predictor State')))
        #list_loss_r.append(pd.DataFrame(x.Scalars('Train/Loss World Predictor Reward')))
        # do some stats stuff
        print(run)
        print(dsi_df[dsi_df["step"] > 900000].describe())
        print(dsns_df[dsns_df["step"] > 900000].describe())
    ##### first plot diff to next state and identity #####
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    # reward for diff to identity
    df1 = pd.concat(list_dsi)
    df2 = pd.concat(list_dsns)
    # smooth
    df1["s-value"] = smooth2(df1["value"], 121)
    df2["s-value"] = smooth2(df2["value"], 121)
    '''
    # plot first
    p = sns.lineplot(data=df1, x="step", y="value", alpha=0.2, legend=False, color="green")
    sns.lineplot(data=df1, x="step", y="s-value", color="green")
    sns.lineplot(data=df2, x="step", y="value", alpha=0.2, legend=False, color="red")
    sns.lineplot(data=df2, x="step", y="s-value", color="red")
    p.set(ylabel='Loss', xlabel='Frame')
    # finish
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    #plt.savefig("dummy.pdf")
    plt.show()
    '''



# call function to plot
plot_mdr()