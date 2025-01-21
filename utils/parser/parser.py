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
    parser.add_argument("--progress", action="store_true", help="display a progress bar of the training process")
    parser.add_argument("--hud", action="store_true", help="use HUD objects")

    opts = parser.parse_args()

    env_str = "ALE/" + opts.game +"-v5"
    settings_str = ""
    pruned_ff_name = None
    hide_properties = False
    noisy = os.environ.get("SCOBI_OBJ_EXTRACTOR", "OC_Atari") == "Noisy_OC_Atari"
    focus_dir = "resources/focusfiles"


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

    return {
        "exp_name": exp_name,
        "env": env_str,
        "hide_properties": hide_properties,
        "pruned_ff_name": pruned_ff_name,
        "focus_dir": focus_dir,
        "reward_mode": reward_mode,
        "rgb_exp": opts.rgb,
        "seed": opts.seed,
        "environments": opts.environments,
        "game": opts.game,
        "rgb": opts.rgb,
        "reward": opts.reward,
        "progress": opts.progress,
        "hud": opts.hud
    }



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
    parser.add_argument("--record", required= False, action="store_true", help="wheter to record the rendered video")
    parser.add_argument("--nb_frames", type=int, default=0, help="stop recording after nb_frames (or 1 episode if not specified)")
    parser.add_argument("--print-reward", action="store_true", help="display the reward in the console (if not 0)")
    parser.add_argument("--viper", nargs="?", const=True, default=False, help="evaluate the extracted viper tree instead of a checkpoint")
    parser.add_argument("--hud", action="store_true", help="use HUD objects")
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


    exp_name = opts.game + "_seed" + str(opts.seed) + settings_str

    return {
        "exp_name": exp_name,
        "env_str": env_str,
        "hide_properties": hide_properties,
        "pruned_ff_name": pruned_ff_name,
        "variant": variant,
        "version": opts.version or -1,
        "game": opts.game,
        "seed": opts.seed,
        "reward": opts.reward,
        "rgb": opts.rgb,
        "record": opts.record,
        "nb_frames": opts.nb_frames,
        "print_reward": opts.print_reward,
        "viper": opts.viper,
        "hud": opts.hud
    }


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
    parser.add_argument("--viper", nargs="?", const=True, default=False, help="evaluate the extracted viper tree instead of a checkpoint")
    parser.add_argument("--hud", action="store_true", help="use HUD objects")
    parser.add_argument("--hackatari", action="store_true", help="use Hackatari as environment")
    parser.add_argument("-mods", "--mods", type=str, required=False, help="list which mods you want to run with hackatari, separate via comma")
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
        
    exp_name = opts.game + "_seed" + str(opts.seed) + settings_str

    if opts.hackatari:
        mods = [item.strip() for item in opts.mods.split(",")]
    else: mods = None

    return {
        "exp_name": exp_name,
        "env_str": env_str,
        "hide_properties": hide_properties,
        "pruned_ff_name": pruned_ff_name,
        "times": opts.times,
        "variant": variant,
        "version": opts.version or -1,
        "game": opts.game,
        "seed": opts.seed,
        "reward": opts.reward,
        "progress": opts.progress,
        "rgb": opts.rgb,
        "viper": opts.viper,
        "hud": opts.hud,
        "hackatari": opts.hackatari,
        "mods": mods
    }


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