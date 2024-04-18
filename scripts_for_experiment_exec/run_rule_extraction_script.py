import subprocess
import os

def run_train_script(eclaire_cfg_file):
    # Define the command and parameters as a list
    command = [
        'python', '-m', 'baselines.rules',
        '-- eclaire-cfg-file', eclaire_cfg_file
    ]

    # Execute the command without capturing the output, so it's displayed in the terminal
    result = subprocess.run(command, text=True)

    # Check if the execution was successful
    if result.returncode == 0:
        print(f"Execution successful for parameters: {command[4:]}")
    else:
        print(f"Execution failed for parameters: {command[4:]}")


params = []
for eclaire_config_file in os.listdir("eclaire_configs"):
    if not eclaire_config_file.endswith(".yaml"):
        continue
    eclaire_cfg_file = os.path.join("eclaire_configs", eclaire_config_file)
    params.append(eclaire_cfg_file)

for param in params:
    run_train_script(param)
