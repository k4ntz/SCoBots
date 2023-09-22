from remix.rules.ruleset import Ruleset
import torch
import math

fnames_skiing = ['Player1.x', 'Player1.y', 'Mogul1.x', 'Mogul1.y', 'Flag1.x', 'Flag1.y', 'Flag2.x', 'Flag2.y', 'Tree1.x', 'Tree1.y', 'Tree2.x', 'Tree2.y', 'O(Player1)', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Flag1.x', 'Flag1.y', 'Flag1.x[t-1]', 'Flag1.y[t-1]', 'D(Player1,Flag1).x', 'D(Player1,Flag1).y', 'C(Flag1,Flag2).x', 'C(Flag1,Flag2).y', 'DV(Player1).x', 'DV(Player1).y', 'DV(Flag1).x', 'DV(Flag1).y']
fnames_pong = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Ball1,Ball1).x', 'LT(Ball1,Ball1).y', 'LT(Ball1,Enemy1).x', 'LT(Ball1,Enemy1).y', 'LT(Ball1,Player1).x', 'LT(Ball1,Player1).y', 'LT(Enemy1,Ball1).x', 'LT(Enemy1,Ball1).y', 'LT(Enemy1,Enemy1).x', 'LT(Enemy1,Enemy1).y', 'LT(Enemy1,Player1).x', 'LT(Enemy1,Player1).y', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'LT(Player1,Enemy1).x', 'LT(Player1,Enemy1).y', 'LT(Player1,Player1).x', 'LT(Player1,Player1).y', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']
fnames_pong_no_enemy = ['Ball1.x', 'Ball1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'DV(Ball1).x', 'DV(Ball1).y', 'DV(Player1).x', 'DV(Player1).y']
fnames_pong_pruned_no_lt = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']

fnames = fnames_pong_pruned_no_lt


GAME_ENV = "pong_enemy"
data_base_path = "remix_data/" + GAME_ENV + "/"

ruleset = Ruleset().from_file("remix_data/" + GAME_ENV + "/output.rules")
checkpoint = torch.load(data_base_path + "checkpoint.pth")
norm_state = checkpoint["normalizer_state"]
denorm_dict = {}

for state, name in zip(norm_state, fnames):
    mean = state["m"]
    variance = state["s"] / (state["n"])
    standard_deviation = math.sqrt(variance)
    denorm_dict[name] = (mean, standard_deviation)



for rule in ruleset:
    for p in rule.premise:
        terms_set = p.terms
        for term in terms_set:
            m, s = denorm_dict[term.variable]
            denorm_value = int((term.threshold * s) + m)
            term.threshold = denorm_value



ruleset.to_file(data_base_path + "denormed.rules")