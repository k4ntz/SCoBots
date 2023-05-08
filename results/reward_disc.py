import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(6,2.5))
fs = 11

df = pd.read_csv("reward_discovery.csv")
scobi = df[0:5]
iscobi = df[6:11]

rewards_thresholds = [1, 3, 5, 10, 15, 20, 25, 30]

scobi_means = []
iscobi_means = []
scobi_stds = []
iscobi_stds = []
for nb_reward in rewards_thresholds:
    scobi_means.append(np.mean(scobi.loc[:,str(nb_reward)]))
    iscobi_means.append(np.mean(iscobi.loc[:,str(nb_reward)]))
    scobi_stds.append(np.std(scobi.loc[:,str(nb_reward)]))
    iscobi_stds.append(np.std(iscobi.loc[:,str(nb_reward)]))
x = np.arange(len(rewards_thresholds))
plt.plot(x, scobi_means, label="Original", color="tab:blue")
plt.errorbar(x, scobi_means, yerr=scobi_stds, color="tab:blue")
plt.plot(x, iscobi_means, label="Assisted", color="tab:orange")
plt.errorbar(x, iscobi_means, yerr=iscobi_stds, color="tab:orange")
plt.legend(fontsize=fs+2)
plt.xlabel("Observed rewards", fontsize=fs)
plt.xticks(x, labels=rewards_thresholds, fontsize=fs)
plt.ylabel("Steps", fontsize=fs)
yticks = ax.get_yticks()
ax.set_yticklabels([f"{int(y/1000)}K" for y in yticks], fontsize=fs); # use LaTeX formatted labels
plt.title("Reward discovery in Pong", fontsize=fs+2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
if "-s" in sys.argv:
    plt.savefig("reward_discovery.pdf")

plt.show()