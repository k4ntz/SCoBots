from remix.rules.ruleset import Ruleset
import torch
import math
from fnames import get_fnames_for_game

GAME_ENV = "freeway"
fnames = get_fnames_for_game(GAME_ENV) #check whether pruned info must be added
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