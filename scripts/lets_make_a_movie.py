import os
import subprocess
import re

def execute(base_dir, script_to_run):
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            match = re.match(
                r"(\w+)_seed(\d+)_reward-([a-z]+)_([a-z]+)(_([a-z]+))?",
                folder_name
            )
            if match:
                game_name = match.group(1)
                seed = match.group(2)
                reward_type = match.group(3)
                reward_detail = match.group(4)
                modifier = match.group(6)

                if reward_detail == "oc":
                    if modifier == "pruned":
                        command = [
                            "python", script_to_run,
                            "-g", game_name,
                            "-s", seed,
                            "-r", reward_type,
                            "-p default",
                            "--record",
                            "--nb_frames", "10"
                        ]
                    else :
                        command = [
                        "python", script_to_run,
                        "-g", game_name,
                        "-s", seed,
                        "-r", reward_type,
                        "--record",
                        "--nb_frames", "10"
                        ]
                else:
                    if modifier == "pruned":
                        command = [
                            "python", script_to_run,
                            "-g", game_name,
                            "-s", seed,
                            "-r", reward_type,
                            "-p default",
                            "--rgb",
                            "--record",
                            "--nb_frames", "10"
                        ]
                    else :
                        command = [
                        "python", script_to_run,
                        "-g", game_name,
                        "-s", seed,
                        "-r", reward_type,
                        "--rgb",
                        "--record",
                        "--nb_frames", "10"
                        ]
                print(f"Executing: {' '.join(command)}")
                subprocess.run(command, check=True, cwd="SET THE WORKING DIR HERE MIGHT NOT WORK OTHERWISE")
            else:
                print(f"Skipping '{folder_name}'")

base_directory = "resources/checkpoints"
script_to_execute = "render_agent.py"

# Run it from the working directory or hardcode the path to the folder/scripts 
execute(base_directory, script_to_execute)
