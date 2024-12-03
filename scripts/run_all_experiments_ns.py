import subprocess

seeds = [0, 1, 2]
experiments = ["Asterix", "Bowling", "Boxing", "Freeway", "Kangaroo", "Pong", "Seaquest", "Skiing", "Tennis"]

# Goal: three running processes (one for each seed, each with different GPU)

def build_command(game, seed):
    reward = "human" if game in ["Skiing", "Kangaroo", "Pong"] else "env"
    hud = game in ["Kangaroo", "Seaquest"]
    # command = f"python run_experiment.py -g {game} -s {seed} -env 8 -r {reward} --progress -p default"
    dev = 15 - seed
    command = [f"CUDA_VISIBLE_DEVICES={dev}", "uv", "run", "train.py", "-g", game, "-s", str(seed), "-env", "8", "-r", reward, "--progress", "-p", "default"]
    if hud:
        command += ["--hud"] 
    return command

commands = []
for e in experiments:
    for s in seeds:
        command = build_command(e, s)
        commands.append(command)

import subprocess
from concurrent.futures import ProcessPoolExecutor
import os

# Function to execute a command
def run_command(command):
    process_id = os.getpid()  # Get the current process ID
    print(f"{process_id}: Running command: ", " ".join(command))
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        return f"Executor PID: {process_id}, Output: {result.stdout.strip()}, Errors: {result.stderr.strip()}"
    except subprocess.CalledProcessError as e:
        return f"Executor PID: {process_id}, Command failed with return code {e.returncode}, Errors: {e.stderr.strip()}"

# Use ProcessPoolExecutor to execute commands with 3 processes
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_command, commands))

# Print results
for i, result in enumerate(results):
    print(f"Command {i + 1}: {result}")
