import imp
import cv2

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import seaborn as sns

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from sklearn.decomposition import PCA

from xrl.utils.utils import features_names
from xrl.utils.utils import get_integrated_gradients

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
    def plot_IG_img(self, ig, exp_name, input, feature_titles, target_class, obs, plot):
        attr = get_integrated_gradients(ig, input, target_class)
        attr_df = pd.DataFrame({"Values": attr},
                      index=feature_titles)
        #print(attr_df)
        env_img = obs
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
        resized = cv2.resize(mat, (240, 240), interpolation = cv2.INTER_AREA)
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