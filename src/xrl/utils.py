from operator import contains
import os
import math
import torch
import cv2
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import seaborn as sns

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import entropy

from argparse import ArgumentParser
from xrl.xrl_config import cfg

# player enemy ball
features_names = [
    "0: player speed",
    "1: x enemy - player",
    "2: y enemy - player",
    "3: x ball - player",
    "4: y ball - player",
    "5: y enemy-target - player",
    "6: x enemy-target - player",
    "7: y ball-target - player",
    "8: x ball-target - player",
    "9: enemy speed",
    "10: x ball - enemy",
    "11: y ball - enemy",
    "12: y player-target - enemy",
    "13: x player-target - enemy",
    "14: y ball-target - enemy",
    "15: x ball-target - enemy",
    "16: ball speed",
    "17: y player-target - ball",
    "18: x player-target - ball",
    "19: y enemy-target - ball",
    "20: x enemy-target - ball",
]

######################
######## INIT ########
######################

# function to get config
def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )

    parser.add_argument(
        '--space-config-file',
        type=str,
        default='configs/atari_ball_joint_v1.yaml',
        metavar='FILE',
        help='Path to SPACE config file'
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')

    # Seed
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return cfg

######################
###### PLOTTING ######
######################

plt.interactive(False)

# function to plot live while training
def plot_screen(env, episode, step, second_img=None):
    plt.figure(3)
    plt.title('Episode: ' + str(episode) + " - Step: " + str(step))
    plt.imshow(env.render(mode='rgb_array'),
           interpolation='none')
    if second_img is not None:
        plt.figure(2)
        plt.clf()
        plt.title('X - Episode: ' + str(episode) + " - Step: " + str(step))
        plt.imshow(second_img)
    plt.plot()
    plt.pause(0.0001)  # pause a bit so that plots are updated


# function to get integrated gradients
def get_integrated_gradients(ig, input, target_class):
    # get attributions and print
    attributions, approximation_error = ig.attribute(input,
        target=target_class, return_convergence_delta=True)
    #print(attributions)
    attr = attributions[0].cpu().detach().numpy()
    #print(attr_df)
    return attr


# function to get feature titles
def get_feature_titles(n_raw_features = 3):
    feature_titles = []
    for i in range(0, n_raw_features):
        feature_titles.append(str("obj" +  str(i) + " vel"))
        for j in range(0, n_raw_features):
            if j > i:
                feature_titles.append(str("x obj" + str(j) + " - obj" + str(i)))
                feature_titles.append(str("y obj" + str(j) + " - obj" + str(i)))
        for j in range(0, n_raw_features):
            if i != j:
                feature_titles.append(str("target y obj" + str(j) + " - obj" + str(i)))
                feature_titles.append(str("target x obj" + str(j) + " - obj" + str(i)))
    return feature_titles


class Plotter():
    def __init__(self, figsize=(20, 10)):
        fig, axes = plt.subplots(ncols=2, figsize=figsize,
                                 gridspec_kw={'width_ratios': [6, 1]})
        self.fig = fig
        self.axes = axes
        self.cbar = True
        # self.fig.tight_layout()

    # helper function to get integrated gradients of given features as plotable image
    def plot_IG_img(self, ig, exp_name, input, feature_titles, target_class, env, plot):
        attr = get_integrated_gradients(ig, input, target_class)
        attr_df = pd.DataFrame({"Values": attr},
                      index=feature_titles)
        #print(attr_df)
        env_img = env.render(mode='rgb_array')
        # plot both next to each other
        ax1, ax2 = self.axes
        ax1.imshow(env_img)
        sn.heatmap(attr_df, ax=ax2, vmin=-0.2, vmax=1, cbar=self.cbar)
        self.cbar = False
        ax1.set_title(exp_name)
        # convert fig to cv2 img
        # put pixel buffer in numpy array
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(mat, (480, 480), interpolation = cv2.INTER_AREA)
        if plot:
            plt.draw()
            plt.pause(0.0001)
        # clean up
        for ax in self.axes:
            ax.clear()
        # self.fig.clf()
        # plt.close(self.fig)
        return resized


# 0: "NOOP",
# 1: "FIRE",
# 2: "UP",
# 3: "RIGHT",
# 4: "LEFT",
# 5: "DOWN",
# helper function to create pcas based on actions
def ig_pca(ig_action_sum, action_meanings):
    plt.figure(figsize=(8,6))
    # process ig df with actions
    mapping = {i: action_meanings[i] for i in range(len(action_meanings))}
    # watch out, n_cols is -1 from col count
    n_cols = len(ig_action_sum[1,:]) - 1
    ig_df = pd.DataFrame(data=ig_action_sum).replace({n_cols: mapping}).rename(columns={n_cols: "action"})
    # pca for ig values
    cut_ig_df = ig_df.iloc[:,:n_cols]
    pca = PCA(n_components=2)
    pca.fit(cut_ig_df)
    x = pca.transform(cut_ig_df)
    plot = sns.scatterplot(x[:,0], x[:,1], hue=ig_df["action"], palette="deep")
    plot.set_title("PCA on Integrated Gradient Values")
    plt.tight_layout()
    plt.show()




# helper function to plot igs of each feature over whole episode
def plot_igs(ig_sum, plot_titles, action_meanings):
    # process ig df with actions
    mapping = {i: action_meanings[i] for i in range(len(action_meanings))}
    # watch out, n_cols is -1 from col count
    n_cols = len(ig_sum[1,:]) - 1
    ig_df = pd.DataFrame(data=ig_sum).replace({n_cols: mapping}).rename(columns={n_cols: "action"})
    # remove action col
    cut_ig_df = ig_df.iloc[:,:n_cols]
    for i, col in enumerate(cut_ig_df):
        igs = cut_ig_df[col]
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        sns.set(font_scale=0.7)
        sns.set_style("ticks")
        ## seaborn lineplot
        scatter = sns.scatterplot(x=igs.index, y=igs, size=0.02, alpha=0.7, palette="deep", hue=ig_df["action"], ax=ax[0])
        scatter.set(xlabel='Step', ylabel='Integrated gradient')
        if plot_titles is not None:
            scatter.set_title("Integrated gradient values for feature: " + plot_titles[i])
        ## seaborn histogram
        hist = sns.histplot(x=igs, hue=ig_df["action"], palette="deep", bins=25, element="bars", fill=False, ax=ax[1])
        if plot_titles is not None:
            hist.set_title("Integrated gradient histogram for feature: " + plot_titles[i])
        fig.tight_layout()
        #plt.savefig("ig_" + plot_titles[i] + ".png", dpi=300)
        plt.show()


# plot correlation matrix with given set of features
def plot_corr(features):
    df = pd.DataFrame(data=features, columns=features_names)
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    plt.figure(figsize=(6,5))
    corrplot = sns.heatmap(
        df.corr(),
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    # finish
    corrplot.set(xticklabels=[])
    corrplot.set(xlabel="Features")
    corrplot.set(yticklabels=[])
    corrplot.set(ylabel="Features")
    #corrplot.tick_params(bottom=False, left=False)  # remove the ticks
    sns.despine(offset=10, trim=True, left=True, bottom=True)
    plt.tight_layout()
    #plt.savefig("corr-features-rand-min.pdf", dpi=300)
    plt.show()
    return None


# plot function for violin plotsw
# helper function to plot igs of each feature over whole episode
def plot_igs_violin(ig_sum, feature_titles, action_meanings):
    # process ig df with actions
    mapping = {i: action_meanings[i] for i in range(len(action_meanings))}
    # watch out, n_cols is -1 from col count
    n_cols = len(ig_sum[1,:]) - 1
    ig_df = pd.DataFrame(data=ig_sum).replace({n_cols: mapping}).rename(columns={n_cols: "action"})
    # rename cols to feature names
    ig_df = ig_df.rename(columns={i : feature_titles[i] for i in range(len(feature_titles))})
    #print(ig_df)
    # set action col as index
    ig_df = ig_df.set_index("action")
    #print(ig_df)
    # convert to action | feature | ig-value
    df = ig_df.stack()
    df = df.reset_index()
    df.columns = ["Action", "Feature", "IG-Value"]
    print(df)
    # norm ig values to -1 | 1
    #df["IG-Value-n"] = (df["IG-Value"] - df["IG-Value"].mean()) / (df["IG-Value"].max() - df["IG-Value"].min())
    # z score normalisation
    #df["IG-Value-n"] = (df["IG-Value"] - df["IG-Value"].mean()) / df["IG-Value"].std()
    print(df)
    # remove rows with action used less than x%
    count = df["Action"].value_counts()
    print(count)
    df = df[df.isin(count.index[count >= len(df.index) * 0.05]).values]
    count2 = df["Action"].value_counts()
    print(count2)
    # plot violin
    plt.figure(figsize=(10,5))
    sns.set(font_scale=0.7)
    sns.set_style("whitegrid")
    plot = sns.violinplot(x="Feature", y="IG-Value", hue="Action", data=df, palette="deep", linewidth=0.5)
    plot.set_title("Integrated gradient values for each feature")
    plot.set(ylabel='IG')
    plt.tight_layout()
    plt.show()



# plot weight of lin model
def plot_lin_weights(model, feature_titles, actions):
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,7))
    # build df
    weights = model.out.weight
    weights_df = pd.DataFrame(weights.numpy(), columns=feature_titles, index=actions)
    weights_df2 = weights_df.stack()
    weights_df2 = weights_df2.reset_index()
    weights_df2.columns = ["Action", "Feature", "Weight"]
    # plot
    ax = sns.boxplot(data=weights_df2, x='Action',y='Weight', color="white")
    ax = sns.stripplot(data=weights_df2, x='Action',y='Weight',hue='Feature', palette="deep", jitter=0, size=6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # finish plot
    sns.despine(offset=1, trim=True, bottom=True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.savefig("weights.pdf")
    plt.show()

###############################
##### PROCESSING FEATURES #####
###############################


# function to get raw features and order them by
def get_raw_features(env_info, last_raw_features=None, gametype=0):
    # extract raw features
    labels = env_info["labels"]
    # if ball game
    if gametype == 0:
        player = [labels["player_x"].astype(np.int16),
                labels["player_y"].astype(np.int16)]
        enemy = [labels["enemy_x"].astype(np.int16),
                labels["enemy_y"].astype(np.int16)]
        ball = [labels["ball_x"].astype(np.int16),
                labels["ball_y"].astype(np.int16)]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, enemy, ball, None, None, None]
        else:
            raw_features = np.roll(raw_features, 3)
            raw_features[0] = player
            raw_features[1] = enemy
            raw_features[2] = ball
        return raw_features
    ###########################################
    # demon attack game
    elif gametype == 1:
        player = [labels["player_x"].astype(np.int16),
                np.int16(3)]        # constant 3
        enemy1 = [labels["enemy_x1"].astype(np.int16),
                labels["enemy_y1"].astype(np.int16)]
        enemy2 = [labels["enemy_x2"].astype(np.int16),
                labels["enemy_y2"].astype(np.int16)]
        enemy3 = [labels["enemy_x3"].astype(np.int16),
                labels["enemy_y3"].astype(np.int16)]
        #missile = [labels["player_x"].astype(np.int16),
        #        labels["missile_y"].astype(np.int16)]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, enemy1, enemy2, enemy3, None, None, None, None]
        else:
            raw_features = np.roll(raw_features, 4)
            raw_features[0] = player
            raw_features[1] = enemy1
            raw_features[2] = enemy2
            raw_features[3] = enemy3
        return raw_features
    ###########################################
    # boxing game
    elif gametype == 2:
        player = [labels["player_x"].astype(np.int16),
                labels["player_y"].astype(np.int16)]
        enemy = [labels["enemy_x"].astype(np.int16),
                labels["enemy_y"].astype(np.int16)]
        # set new raw_features
        raw_features = last_raw_features
        if raw_features is None:
            raw_features = [player, enemy, None, None]
        else:
            raw_features = np.roll(raw_features, 2)
            raw_features[0] = player
            raw_features[1] = enemy
        return raw_features



# helper function to calc linear equation
def get_lineq_param(obj1, obj2):
    x = obj1
    y = obj2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


# helper function to convert env info into custom list
# raw_features contains player x, y, ball x, y, oldplayer x, y, oldball x, y,
# features are processed stuff for policy
def preprocess_raw_features(raw_features):
    n_raw_features = int(len(raw_features)/2)
    features = []
    for i in range(0, n_raw_features):
        obj1, obj1_past = raw_features[i], raw_features[i + n_raw_features]
        # when object has moved and has history
        if obj1_past is not None and not (obj1[0] == obj1_past[0] and obj1[1] == obj1_past[1]):
            # append velocity of itself
            features.append(math.sqrt((obj1_past[0] - obj1[0])**2 + (obj1_past[1] - obj1[1])**2))
        else:
            features.append(0)
        for j in range(0, n_raw_features):
            # apped all manhattan distances to all other objects
            # which are not already calculated
            if j > i:
                obj2 = raw_features[j]
                # append coord distances
                features.append(obj2[0] - obj1[0]) # append x dist
                features.append(obj2[1] - obj1[1]) # append y dist
        for j in range(0, n_raw_features):
            # calculate movement paths of all other objects
            # and calculate distance to its x and y intersection
            if i != j:
                obj2, obj2_past = raw_features[j], raw_features[j + n_raw_features]
                # if other object has moved
                if obj2_past is not None and not (obj2[0] == obj2_past[0] and obj2[1] == obj2_past[1]):
                    # append trajectory cutting points
                    m, c = get_lineq_param(obj2, obj2_past)
                    # now calc target pos
                    # y = mx + c substracted from its y pos
                    features.append(np.int16(m * obj1[0] + c) - obj1[1])
                    # x = (y - c)/m substracted from its x pos
                    features.append(np.int16((obj1[1] - c) / m)  - obj1[0])
                else:
                    features.append(0)
                    features.append(0)
    return raw_features, features


# helper function to get features
def do_step(env, action=1, last_raw_features=None):
    obs, reward, done, info = env.step(action)
    # check if ball game
    # 0: ball game
    # 1: demonattack
    # 2: boxing
    name = env.unwrapped.spec.id
    gametype = 0
    if "Demon" in name:
        gametype = 1
    elif "Boxing" in name:
        gametype = 2
    # calculate meaningful features from given raw features and last raw features
    raw_features = get_raw_features(info, last_raw_features, gametype=gametype)
    raw_features, features = preprocess_raw_features(raw_features)
    return raw_features, features, reward, done


# do 5 episodes and get features to prune from corr matrix
def init_corr_prune(env, it = 5, tr = 0.75):
    features_list = []
    # run it episodes to collect features data
    for i in tqdm(range(it)):
        n_actions = env.action_space.n
        _ = env.reset()
        _, _, done, _ = env.step(1)
        raw_features, features, _, _ = do_step(env)
        for t in range(50000):  # max 50k steps per episode
            action = np.random.randint(0, n_actions)
            raw_features, features, reward, done = do_step(env, action, raw_features)
            features_list.append(features)
            if done:
                break
    # now make corr matrix
    df = pd.DataFrame(data=features_list)
    corr = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # Find features with correlation greater than tr
    to_drop = [column for column in upper.columns if any(upper[column] > tr)]
    return to_drop


# calc single entropy of one feature
def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)


# calc entropy of all features given
def calc_entropies(features):
    features = np.array(features)
    entropies = []
    for col in features.T:
        entropies.append(entropy1(col))
    print(entropies)
    return entropies
