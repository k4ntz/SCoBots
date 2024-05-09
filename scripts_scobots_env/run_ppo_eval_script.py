import subprocess

def run_save_ppo_model_script(game, seed, times, layers, input_data, prune):
    # Define the command and parameters as a list
    command = [
        'python', '-m', 'baselines.eval',
        '-g', game,
        '-s', str(seed),
        '-t', str(times),
        '--input_data', input_data,
        '--num_layers', str(layers),
        '--prune', prune,
        '--save_model'
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
    games = ["Pong",]
    seeds = [42]
    cores = [8,]
    times = [5]
    layers = [2, 1]
    #input_datas = ["OCAtari","SPACE",]
    input_datas = ["SPACE",]
    prunes = ["no_prune", "external"]
    adam_step_sizes = [0.001]

    # Loop over each combination of parameters and call the function
    for game in games:
        for seed in seeds:
            for time in times:
                for layer in layers:
                    for input_data in input_datas:
                        for prune in prunes:
                            run_save_ppo_model_script(game, seed, time, layer, input_data, prune)
