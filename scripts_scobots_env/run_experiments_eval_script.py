import subprocess
import os

def run_collect_obs_data(config_file, eclaire_cfg_file, rl_algo):
    # Define the command and parameters as a list

    command = [
        'python', '-m', 'eval',
        '--config-file', config_file,
        '--eclaire-cfg-file', eclaire_cfg_file,
        'rl_algo', str(rl_algo)
    ]
    #python -m eval --config-file configs/re-pong.yaml --eclaire-cfg-file ../eclaire_configs/config_eclaire_Pong_s42_re_pr-nop_OCAtariinput_1l-v3.yaml rl_algo 3
    # Execute the command without capturing the output, so it's displayed in the terminal
    result = subprocess.run(command, text=True)

    # Check if the execution was successful
    if result.returncode == 0:
        print(f"Execution successful for parameters: {command[4:]}")
    else:
        print(f"Execution failed for parameters: {command[4:]}")


if __name__ == "__main__":
    # Define different sets of parameters
    params = []
    dummy_config_file = os.path.join("configs", "re-pong.yaml") #TODO check whether file really does not matter
    for eclaire_config_file in os.listdir("eclaire_configs"):
        if not eclaire_config_file.endswith(".yaml"):
            continue
        if not "Boxing" in eclaire_config_file:
            continue
        if "SPACE" in eclaire_config_file:
            continue
        rl_algo = 3 # encodes PPO
        eclaire_cfg_file = os.path.join("..","eclaire_configs", eclaire_config_file)
        params.append((dummy_config_file, eclaire_cfg_file, rl_algo))

    os.chdir("experiments")

    # Iterate over the different sets of parameters
    for param in params:
        run_collect_obs_data(*param)


