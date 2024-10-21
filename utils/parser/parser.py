import os
import re
from pathlib import Path
import argparse


def parse_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-env", "--environments", type=int, required=True,
                        help="number of envs used")
    parser.add_argument("-r", "--reward", type=str, required=False, choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, required=False, choices=["default", "external"],
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'resources/focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-x", "--exclude_properties", action="store_true", help="exclude properties from feature vector")
    parser.add_argument("--rgb", action="store_true", help="rgb observation space")
    parser.add_argument("--normalize", action="store_true", help="normalizes the observations at each step")
    parser.add_argument("--hud", action="store_true", help="allow agent to access HUD elements")
    parser.add_argument("--progress", action="store_true", help="display a progress bar of the training process")
    opts = parser.parse_args()

    env_str = "ALE/" + opts.game +"-v5"
    settings_str = ""
    pruned_ff_name = None
    hide_properties = False
    focus_dir = "resources/focusfiles"
    noisy = os.environ["SCOBI_OBJ_EXTRACTOR"] == "Noisy_OC_Atari"


    reward_mode = 0
    if opts.reward == "env":
        settings_str += "_reward-env"
    if opts.reward == "human":
        settings_str += "_reward-human"
        reward_mode = 1
    if opts.reward == "mixed":
        settings_str += "_reward-mixed"
        reward_mode = 2

    game_id = env_str.split("/")[-1].lower().split("-")[0]

    #override some settings if rgb
    rgb_exp = opts.rgb
    if opts.rgb:
        settings_str += "_rgb"
    else:
        settings_str += "_oc"

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
    if opts.prune == "default":
        settings_str += "_pruned"
    if opts.prune == "external":
        settings_str += "_pruned-external"
        focus_dir = "resources/focusfiles"
    if opts.exclude_properties:
        settings_str += '_excludeproperties'
        hide_properties = True

    exp_name = opts.game + "_seed" + str(opts.seed) + settings_str
    if noisy:
        exp_name += "-noisy"


    return exp_name, env_str, hide_properties, pruned_ff_name, focus_dir, reward_mode, rgb_exp, opts.seed, opts.environments, opts.game, opts.rgb, opts.reward, opts.normalize, opts.hud, opts.progress



def render_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-r", "--reward", type=str, required=False, choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, required=False, choices=["default", "external"],
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'resources/focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-x", "--exclude_properties",  action="store_true", help="exclude properties from feature vector")
    parser.add_argument("-n", "--version", type=str, required=False, help="specify which trained version. standard selects highest number")
    parser.add_argument("--rgb", required= False, action="store_true", help="rgb observation space")
    parser.add_argument("--normalize", action="store_true", help="normalizes the observations at each step")
    parser.add_argument("--hud", action="store_true", help="allow agent to access HUD elements")

    opts = parser.parse_args()
    parser.add_argument("--record", required= False, action="store_true", help="wheter to record the rendered video")
    parser.add_argument("--nb_frames", type=int, default=0, help="stop recording after nb_frames (or 1 episode if not specified)")
    parser.add_argument("--print-reward", action="store_true", help="display the reward in the console (if not 0)")
    return parser.parse_args()


def convert_args(opts):
    env_str = "ALE/" + opts.game +"-v5"
    settings_str = ""
    pruned_ff_name = None
    hide_properties = False
    variant = "scobots"

    if opts.reward == "env":
        settings_str += "_reward-env"
    if opts.reward == "human":
        settings_str += "_reward-human"
    if opts.reward == "mixed":
        settings_str += "_reward-mixed"

    game_id = env_str.split("/")[-1].lower().split("-")[0]


    if opts.rgb:
        settings_str += "_rgb"
        variant= "rgb"
    else:
        settings_str += "_oc"

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
        variant =  "iscobots"
    if opts.prune == "default":
        settings_str += "_pruned"
    if opts.prune == "external":
        settings_str += "_pruned-external"
    if opts.exclude_properties:
        settings_str += '_excludeproperties'
        hide_properties = True


    if opts.version:
        version = opts.version
    else:
        version = 0
    exp_name = opts.game + "_seed" + str(opts.seed) + settings_str

    return exp_name, env_str, hide_properties, pruned_ff_name, variant, version, opts.normalize, opts.hud


def parse_eval(parser):
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-t", "--times", type=int, required=True,
                        help="number of episodes to eval")
    parser.add_argument("-r", "--reward", type=str, required=False, choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, required=False, choices=["default", "external"],
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'resources/focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-x", "--exclude_properties", action="store_true", help="exclude properties from feature vector")
    parser.add_argument("-n", "--version", type=str, required=False, help="specify which trained version. standard selects highest number")
    parser.add_argument("--progress", action="store_true", help="display a progress bar of the training process")
    parser.add_argument("--rgb", required= False, action="store_true", help="rgb observation space")
    parser.add_argument("--normalize", action="store_true", help="normalizes the observations at each step")
    parser.add_argument("--hud", action="store_true", help="allow agent to access HUD elements")
    parser.add_argument("--viper", nargs="?", const=True, default=False, help="evaluate the extracted viper tree instead of a checkpoint")
    opts = parser.parse_args()

    env_str = "ALE/" + opts.game +"-v5"
    settings_str = ""
    pruned_ff_name = None
    hide_properties = False
    variant = "scobots"

    if opts.reward == "env":
        settings_str += "_reward-env"
    if opts.reward == "human":
        settings_str += "_reward-human"
    if opts.reward == "mixed":
        settings_str += "_reward-mixed"

    game_id = env_str.split("/")[-1].lower().split("-")[0]


    #override setting str if rgb
    if opts.rgb:
        settings_str += "_rgb"
        variant= "rgb"
    else:
        settings_str += "_oc"

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
        variant =  "iscobots"
    if opts.prune == "default":
        settings_str += "_pruned"
    if opts.prune == "external":
        settings_str += "_pruned-external"
    if opts.exclude_properties:
        settings_str += '_excludeproperties'
        hide_properties = True
        
    exp_name = opts.game + "_seed" + str(opts. seed) + settings_str

    if opts.version:
        version = opts.version
    else:
        version = 0

    return exp_name, env_str, hide_properties, pruned_ff_name, opts.times, variant, version, opts.progress, opts.viper, opts.normalize, opts.hud


def get_highest_version(agent): 
    version = "-n"
    full_path = Path(agent)
    exp_name = full_path.name  # Extract the experiment name (e.g., 'something')

    highest_version = 1
    version_pattern = re.compile(rf"{exp_name}-n(\d+)")  # Regex to match 'exp_name-versionX'

    for directory in Path("resources/checkpoints").iterdir():
        if directory.is_dir():  # Ensure it's a directory
            match = version_pattern.match(directory.name)
            if match:
                version_num = int(match.group(1))  # Extract the version number
                highest_version = max(highest_version, version_num)
    if highest_version > 1:
        return version + str(highest_version)
    else:
        return ""