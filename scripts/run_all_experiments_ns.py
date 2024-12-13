import subprocess
import os

seeds = [0, 1, 2]
experiments = ["Asterix", "Bowling", "Boxing", "Freeway", "Kangaroo", "Pong", "Seaquest", "Skiing", "Tennis"]

# Goal: three running processes (one for each seed, each with different GPU)

def build_train_command(game, seed):
    reward = "human" if game in ["Skiing", "Kangaroo", "Pong"] else "env"
    hud = game in ["Kangaroo", "Seaquest"]
    # command = f"python run_experiment.py -g {game} -s {seed} -env 8 -r {reward} --progress -p default"
    dev = 11 - seed
    command= ["uv", "run", "train.py", "-g", game, "-s", str(seed), "-env", "8", "-r", reward, "--progress", "-p", "default"]
    if hud:
        command += ["--hud"]
    command_dev = (command, dev)
    return command_dev

def build_viper_command(game, seed):
    reward = "human" if game in ["Skiing", "Kangaroo", "Pong"] else "env"
    dev = 11 - seed
    game_string = game + "_seed" + str(seed) + "_reward-" + reward + "_oc_pruned" 
    command= ["uv", "run", "viper_extract.py", "-i", game_string, "-r", "viper"]
    command_dev = (command, dev)
    return command_dev


# Function to execute a single command
def run_command(command_dev):
    process_id = os.getpid()  # Get the current process ID
    command = command_dev[0]
    device = command_dev[1]
    print(f"Executor PID: {process_id}, device: {device}, command: {command}")
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout, stderr = process.communicate()  # Wait for the process to finish
        if process.returncode == 0:
            return f"Executor PID: {process_id}, Output: {stdout.strip()}, Errors: {stderr.strip()}"
        else:
            return f"Executor PID: {process_id}, Command failed with return code {process.returncode}, Errors: {stderr.strip()}"
    except Exception as e:
        return f"Executor PID: {process_id}, Command failed with error: {str(e)}"

# Start three separate processes, each working through its assigned commands
if __name__ == "__main__":
    command_devs = []
    devices = []
    for e in experiments:
        for s in seeds:
            command_dev = build_viper_command(e, s)
            command_devs.append(command_dev)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run_command, command_devs))

    # Print results
    for i, result in enumerate(results):
        print(f"Command {i + 1}: {result}")