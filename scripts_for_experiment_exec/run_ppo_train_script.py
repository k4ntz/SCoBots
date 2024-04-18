import subprocess

def run_train_script(game, seed, cores, layers, input_data, adam_step_size, prune):
    # Define the command and parameters as a list
    command = [
        'python', '-m', 'baselines.train',
        '-g', game,
        '-s', str(seed),
        '-c', str(cores),
        '--num_layers', str(layers),
        '--input_data', input_data,
        '--adam_step_size', str(adam_step_size),
        '--prune', prune
    ]

    # Execute the command without capturing the output, so it's displayed in the terminal
    result = subprocess.run(command, text=True)

    # Check if the execution was successful
    if result.returncode == 0:
        print(f"Execution successful for parameters: {command[4:]}")
    else:
        print(f"Execution failed for parameters: {command[4:]}")

# Define different sets of parameters
games = ["Pong",]
seeds = [42]
cores = [8,]
#layers = [1,2]
layers = [1,]
#input_datas = ["OCAtari","SPACE",]
input_datas = ["OCAtari",]
#prunes = ["no_prune", "external"]
prunes = ["no_prune",]
adam_step_sizes = [0.001]

# Loop over each combination of parameters and call the function
for game in games:
    for seed in seeds:
        for core in cores:
            for layer in layers:
                for input_data in input_datas:
                    for adam_step_size in adam_step_sizes:
                        for prune in prunes:
                            run_train_script(game, seed, core, layer, input_data, adam_step_size, prune)
