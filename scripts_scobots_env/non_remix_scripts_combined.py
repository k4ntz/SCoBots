import subprocess
import os
from run_ppo_train_script import run_ppo_train_script
from run_ppo_eval_script import run_save_ppo_model_script
from run_ppo_save_eval_results import run_ppo_save_eval_results
from run_experiments_eval_script import run_collect_obs_data
#import time

# # sleep for 1,5 hours but print out the time every 10 minutes
# for i in range(9):
#     print(time.ctime())
#     time.sleep(600)


# general parameters
games = ["Boxing",]
layers = [1,2] # ,2
input_datas = ["SPACE",] #"SPACE"
prunes = ["external", "no_prune"] #"no_prune", "external"
seeds = [43]


# training parameters
cores = [8,]
adam_step_sizes = [0.001]

# evaluation parameters
times = [5]

# Loop over each combination of parameters and call the function
for game in games:
    for seed in seeds:
        for core in cores:
            for layer in layers:
                for input_data in input_datas:
                    for adam_step_size in adam_step_sizes:
                        for prune in prunes:
                            run_ppo_train_script(game, seed, core, layer, input_data, adam_step_size, prune)

# Loop over each combination of parameters and call the function
for game in games:
    for seed in seeds:
        for time in times:
            for layer in layers:
                for input_data in input_datas:
                    for prune in prunes:
                        run_save_ppo_model_script(game, seed, time, layer, input_data, prune)

for game in games:
    for seed in seeds:
        for time in times:
            for layer in layers:
                for input_data in input_datas:
                    for prune in prunes:
                        run_ppo_save_eval_results(game, seed, time, layer, input_data, prune)

# params = []
# for game in games:
#     dummy_config_file = os.path.join("configs", f"re-{game.lower()}.yaml") #TODO check whether file really does not matter
#     for eclaire_config_file in os.listdir("eclaire_configs"):
#         if not eclaire_config_file.endswith(".yaml"):
#             continue
#         if not game in eclaire_config_file:
#             continue
#         rl_algo = 3 # encodes PPO
#         eclaire_cfg_file = os.path.join("..","eclaire_configs", eclaire_config_file)
#         params.append((dummy_config_file, eclaire_cfg_file, rl_algo))

# os.chdir("experiments")

# # Iterate over the different sets of parameters
# for param in params:
#     run_collect_obs_data(*param)

