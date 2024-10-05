import json
import numpy as np
import pandas as pd
import sys


from_rainbow = pd.read_csv('from_rainbow.tsv', sep='\t').set_index('Game')
from_rainbow.index = from_rainbow.index.str.capitalize()

from_ddqn = pd.read_csv('from_ddqn.tsv', sep='\t').set_index('Game')

random = from_ddqn['Random']
human = from_ddqn['Human']
dqn = from_ddqn['DQN']
rainbow = from_rainbow['Rainbow']



results = json.load(open("results.json", "r"))
games = []
rewards_games_base = []
rewards_games_pruned = []
# rewards_games_pruned = []
# for game, gresults in results.items():
#     game_name = game.replace("Deterministic-v4", "")
#     games.append(game_name)
#     rewards_base = np.array([val['reward'] for name, val in gresults.items() if not "pruned" in name])
#     rewards_pruned = np.array([val['reward'] for name, val in gresults.items() if "pruned" in name])
#     if len(rewards_base) > 0:
#         rewards_games_base.append(rewards_base)
#     else:
#         rewards_games_base.append(np.array([0]))
#     if len(rewards_pruned) > 0:
#         rewards_games_pruned.append(rewards_pruned)
#     else:
#         rewards_games_pruned.append(np.array([0]))

methods = ["SCoBot", "iScobot"]

def format_text(seeded_res):
    return f"${seeded_res.mean():.1f}\mpm{{{seeded_res.std():.1f}}}$"

for game, gresults in results.items():
    game_name = game.replace("Deterministic-v4", "")
    games.append(game_name)
    rewards_base = np.array([val['reward'] for name, val in gresults.items() if not "pruned" in name])
    rewards_pruned = np.array([val['reward'] for name, val in gresults.items() if "pruned" in name])
    if "-hn" in sys.argv:
        h_score = human[game_name]
        r_score = random[game_name]
        rewards_base = 100*(rewards_base - r_score)/(h_score - r_score)
        rewards_pruned = 100*(rewards_pruned - r_score)/(h_score - r_score)
    if len(rewards_base) > 0:
        rewards_games_base.append(format_text(rewards_base))
    else:
        rewards_games_base.append("-")
    if len(rewards_pruned) > 0:
        rewards_games_pruned.append(format_text(rewards_pruned))
    else:
        rewards_games_pruned.append("-")

df = pd.DataFrame([rewards_games_base, rewards_games_pruned], columns=games, index=methods).T
df.index.name='Game'



if "-hn" in sys.argv:
    rainbow = rainbow[games]
    dqn = dqn[games]
    random = random[games]
    human = human[games]
    rainbow = rainbow.map(lambda x: x.replace(",", "")).astype(float)
    rainbow = round(100 * (rainbow - random)/(human - random), 1)
    # dqn = dqn.map(lambda x: x.replace(",", "")).astype(float)
    dqn = round(100 * (dqn - random)/(human - random), 1)
    df = pd.concat([df, dqn, rainbow], axis=1).rename(columns={0:"DQN" , 1: "Rainbow"})
else:
    df = pd.concat([df, rainbow[games], random[games], human[games]], axis=1)
# dqn_res = dqn_res[games]
# df = pd.concat([df, dqn_res], axis=1)
print(df)
df.to_csv("scores.csv")