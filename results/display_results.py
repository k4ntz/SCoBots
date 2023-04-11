import json
import numpy as np
import pandas as pd

results = json.load(open("results.json", "r"))
from_rainbow = pd.read_csv('from_rainbow.tsv', sep='\t').set_index('Game')
from_rainbow.index = from_rainbow.index.str.capitalize()
rb_res = from_rainbow['Rainbow']
# dqn_res = from_rainbow['DQN']
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

rb_res = rb_res[games]
df = pd.concat([df, rb_res], axis=1)
# dqn_res = dqn_res[games]
# df = pd.concat([df, dqn_res], axis=1)
print(df)
