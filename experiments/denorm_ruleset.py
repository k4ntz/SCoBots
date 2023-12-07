from remix.rules.ruleset import Ruleset
import torch
import math

fnames_skiing = ['Player1.x', 'Player1.y', 'Mogul1.x', 'Mogul1.y', 'Flag1.x', 'Flag1.y', 'Flag2.x', 'Flag2.y', 'Tree1.x', 'Tree1.y', 'Tree2.x', 'Tree2.y', 'O(Player1)', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Flag1.x', 'Flag1.y', 'Flag1.x[t-1]', 'Flag1.y[t-1]', 'D(Player1,Flag1).x', 'D(Player1,Flag1).y', 'C(Flag1,Flag2).x', 'C(Flag1,Flag2).y', 'DV(Player1).x', 'DV(Player1).y', 'DV(Flag1).x', 'DV(Flag1).y']
fnames_pong = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Ball1,Ball1).x', 'LT(Ball1,Ball1).y', 'LT(Ball1,Enemy1).x', 'LT(Ball1,Enemy1).y', 'LT(Ball1,Player1).x', 'LT(Ball1,Player1).y', 'LT(Enemy1,Ball1).x', 'LT(Enemy1,Ball1).y', 'LT(Enemy1,Enemy1).x', 'LT(Enemy1,Enemy1).y', 'LT(Enemy1,Player1).x', 'LT(Enemy1,Player1).y', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'LT(Player1,Enemy1).x', 'LT(Player1,Enemy1).y', 'LT(Player1,Player1).x', 'LT(Player1,Player1).y', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']
fnames_pong_no_enemy = ['Ball1.x', 'Ball1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'DV(Ball1).x', 'DV(Ball1).y', 'DV(Player1).x', 'DV(Player1).y']
fnames_pong_pruned_no_lt = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']
fnames_freeway = ['Chicken1.x', 'Chicken1.y', 'Car1.x', 'Car1.y', 'Car2.x', 'Car2.y', 'Car3.x', 'Car3.y', 'Car4.x', 'Car4.y', 'Chicken1.x', 'Chicken1.y', 'Chicken1.x[t-1]', 'Chicken1.y[t-1]', 'Car1.x', 'Car1.y', 'Car1.x[t-1]', 'Car1.y[t-1]', 'Car2.x', 'Car2.y', 'Car2.x[t-1]', 'Car2.y[t-1]', 'Car3.x', 'Car3.y', 'Car3.x[t-1]', 'Car3.y[t-1]', 'Car4.x', 'Car4.y', 'Car4.x[t-1]', 'Car4.y[t-1]', 'LT(Chicken1,Chicken1).x', 'LT(Chicken1,Chicken1).y', 'LT(Chicken1,Car1).x', 'LT(Chicken1,Car1).y', 'LT(Chicken1,Car2).x', 'LT(Chicken1,Car2).y', 'LT(Chicken1,Car3).x', 'LT(Chicken1,Car3).y', 'LT(Chicken1,Car4).x', 'LT(Chicken1,Car4).y', 'LT(Car1,Chicken1).x', 'LT(Car1,Chicken1).y', 'LT(Car1,Car1).x', 'LT(Car1,Car1).y', 'LT(Car1,Car2).x', 'LT(Car1,Car2).y', 'LT(Car1,Car3).x', 'LT(Car1,Car3).y', 'LT(Car1,Car4).x', 'LT(Car1,Car4).y', 'LT(Car2,Chicken1).x', 'LT(Car2,Chicken1).y', 'LT(Car2,Car1).x', 'LT(Car2,Car1).y', 'LT(Car2,Car2).x', 'LT(Car2,Car2).y', 'LT(Car2,Car3).x', 'LT(Car2,Car3).y', 'LT(Car2,Car4).x', 'LT(Car2,Car4).y', 'LT(Car3,Chicken1).x', 'LT(Car3,Chicken1).y', 'LT(Car3,Car1).x', 'LT(Car3,Car1).y', 'LT(Car3,Car2).x', 'LT(Car3,Car2).y', 'LT(Car3,Car3).x', 'LT(Car3,Car3).y', 'LT(Car3,Car4).x', 'LT(Car3,Car4).y', 'LT(Car4,Chicken1).x', 'LT(Car4,Chicken1).y', 'LT(Car4,Car1).x', 'LT(Car4,Car1).y', 'LT(Car4,Car2).x', 'LT(Car4,Car2).y', 'LT(Car4,Car3).x', 'LT(Car4,Car3).y', 'LT(Car4,Car4).x', 'LT(Car4,Car4).y', 'D(Chicken1,Car1).x', 'D(Chicken1,Car1).y', 'D(Chicken1,Car2).x', 'D(Chicken1,Car2).y', 'D(Chicken1,Car3).x', 'D(Chicken1,Car3).y', 'D(Chicken1,Car4).x', 'D(Chicken1,Car4).y', 'D(Car1,Chicken1).x', 'D(Car1,Chicken1).y', 'D(Car1,Car2).x', 'D(Car1,Car2).y', 'D(Car1,Car3).x', 'D(Car1,Car3).y', 'D(Car1,Car4).x', 'D(Car1,Car4).y', 'D(Car2,Chicken1).x', 'D(Car2,Chicken1).y', 'D(Car2,Car1).x', 'D(Car2,Car1).y', 'D(Car2,Car3).x', 'D(Car2,Car3).y', 'D(Car2,Car4).x', 'D(Car2,Car4).y', 'D(Car3,Chicken1).x', 'D(Car3,Chicken1).y', 'D(Car3,Car1).x', 'D(Car3,Car1).y', 'D(Car3,Car2).x', 'D(Car3,Car2).y', 'D(Car3,Car4).x', 'D(Car3,Car4).y', 'D(Car4,Chicken1).x', 'D(Car4,Chicken1).y', 'D(Car4,Car1).x', 'D(Car4,Car1).y', 'D(Car4,Car2).x', 'D(Car4,Car2).y', 'D(Car4,Car3).x', 'D(Car4,Car3).y', 'V(Chicken1).x', 'V(Car1).x', 'V(Car2).x', 'V(Car3).x', 'V(Car4).x']
fnames_bowling = ['Player1.x', 'Player1.y', 'Pin1.x', 'Pin1.y', 'Pin2.x', 'Pin2.y', 'Pin3.x', 'Pin3.y', 'Pin4.x', 'Pin4.y', 'Ball1.x', 'Ball1.y', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Pin1.x', 'Pin1.y', 'Pin1.x[t-1]', 'Pin1.y[t-1]', 'Pin2.x', 'Pin2.y', 'Pin2.x[t-1]', 'Pin2.y[t-1]', 'Pin3.x', 'Pin3.y', 'Pin3.x[t-1]', 'Pin3.y[t-1]', 'Pin4.x', 'Pin4.y', 'Pin4.x[t-1]', 'Pin4.y[t-1]', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'ED(Player1,Pin1)', 'ED(Player1,Pin2)', 'ED(Player1,Pin3)', 'ED(Player1,Pin4)', 'ED(Player1,Ball1)', 'ED(Pin1,Player1)', 'ED(Pin1,Pin2)', 'ED(Pin1,Pin3)', 'ED(Pin1,Pin4)', 'ED(Pin1,Ball1)', 'ED(Pin2,Player1)', 'ED(Pin2,Pin1)', 'ED(Pin2,Pin3)', 'ED(Pin2,Pin4)', 'ED(Pin2,Ball1)', 'ED(Pin3,Player1)', 'ED(Pin3,Pin1)', 'ED(Pin3,Pin2)', 'ED(Pin3,Pin4)', 'ED(Pin3,Ball1)', 'ED(Pin4,Player1)', 'ED(Pin4,Pin1)', 'ED(Pin4,Pin2)', 'ED(Pin4,Pin3)', 'ED(Pin4,Ball1)', 'ED(Ball1,Player1)', 'ED(Ball1,Pin1)', 'ED(Ball1,Pin2)', 'ED(Ball1,Pin3)', 'ED(Ball1,Pin4)']
fnames = fnames_bowling


GAME_ENV = "bowling"
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