import subprocess
import os

def run_ppo_eval_on_SPACE_and_save(game, seed, times, layers, input_data, prune):
    # Define the command and parameters as a list
    command = [
        'python', '-m', 'baselines.eval',
        '-g', game,
        '-s', str(seed),
        '-t', str(times),
        '--input_data', input_data,
        '--num_layers', str(layers),
        '--prune', prune,
        '--eval_on_SPACE'
    ]

    # Execute the command without capturing the output, so it's displayed in the terminal
    result = subprocess.run(command, text=True)

    # Check if the execution was successful
    if result.returncode == 0:
        print(f"Execution successful for parameters: {command[4:]}")
    else:
        print(f"Execution failed for parameters: {command[4:]}")

if __name__ == "__main__":
    # Define different sets of parameters
    games = ["Pong","Boxing"]
    seeds = [42]
    cores = [8,]
    times = [5]
    layers = [2, 1]
    #input_datas = ["OCAtari","SPACE",]
    input_datas = ["OCAtari",]
    prunes = ["no_prune", "external"]

    # Loop over each combination of parameters and call the function
    for game in games:
        for seed in seeds:
            for time in times:
                for layer in layers:
                    for input_data in input_datas:
                        for prune in prunes:
                            run_ppo_eval_on_SPACE_and_save(game, seed, time, layer, input_data, prune)