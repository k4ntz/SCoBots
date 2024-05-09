import subprocess
import os

def run_train_script(config_file, eclaire_cfg_file, rl_algo):
    # Define the command and parameters as a list

    command = [
        'python', '-m', 'explain',
        '--config-file', config_file,
        '--eclaire-cfg-file', eclaire_cfg_file,
        'rl_algo', str(rl_algo),
        
    ]
    # Execute the command without capturing the output, so it's displayed in the terminal
    result = subprocess.run(command, text=True)

    # Check if the execution was successful
    if result.returncode == 0:
        print(f"Execution successful for parameters: {command[4:]}")
    else:
        print(f"Execution failed for parameters: {command[4:]}")



games = ["Boxing"]
params = []
# Define different sets of parameters
for game in games:
    dummy_config_file = os.path.join("configs", f"re-{game.lower()}.yaml") #TODO check whether file really does not matter
    for eclaire_config_file in os.listdir("eclaire_configs"):
        if not eclaire_config_file.endswith(".yaml"):
            continue
        if game not in eclaire_config_file:
            continue
        if not "SPACE" in eclaire_config_file:
            continue
        print(eclaire_config_file)
        rl_algo = 3 # encodes PPO
        eclaire_cfg_file = os.path.join("..","eclaire_configs", eclaire_config_file)
        params.append((dummy_config_file, eclaire_cfg_file, rl_algo))

os.chdir("experiments")

# Iterate over the different sets of parameters
for param in params:
    run_train_script(*param)


