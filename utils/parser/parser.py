import os
import re
from pathlib import Path


def parse_train(parser):
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
    parser.add_argument("--rgbv4", action="store_true", help="rgb observation space")
    parser.add_argument("--rgbv5", action="store_true", help="rgb observation space")
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

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
    if opts.prune == "default":
        settings_str += "_pruned-default"
    if opts.prune == "external":
        settings_str += "_pruned-external"
        focus_dir = "resources/focusfiles"
    if opts.exclude_properties:
        settings_str += '_excludeproperties'
        hide_properties = True

    if opts.rgbv4 and opts.rgbv5:
        print("please select only one rgb mode!")

    #override some settings if rgb
    rgb_exp = opts.rgbv4 or opts.rgbv5
    if opts.rgbv4:
        settings_str = "-rgbv4"
        env_str = opts.game + "NoFrameskip-v4"
    if opts.rgbv5:
        settings_str = "-rgbv5"

    exp_name = opts.game + "_seed" + str(opts.seed) + settings_str
    if noisy:
        exp_name += "-noisy"


    return exp_name, env_str, hide_properties, pruned_ff_name, focus_dir, reward_mode, rgb_exp, opts.seed, opts.environments, opts.game, opts.rgbv4, opts.rgbv5, opts.reward



def parse_render(parser):
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="seed")
    parser.add_argument("-r", "--reward", type=str, required=False, choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    parser.add_argument("-p", "--prune", type=str, required=False, choices=["default", "external"],
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'resources/focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-x", "--exclude_properties",  action="store_true", help="exclude properties from feature vector")
    parser.add_argument("-v", "--version", type=str, required=False, help="specify which trained version. standard selects highest number")
    parser.add_argument("--rgb", required= False, choices=["rgbv4", "rgbv5"], help="rgb observation space")
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

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
        variant =  "iscobots"
    if opts.prune == "default":
        settings_str += "_pruned-default"
    if opts.prune == "external":
        settings_str += "_pruned-external"
    if opts.exclude_properties:
        settings_str += '_excludeproperties'
        hide_properties = True

    if opts.rgb:
        settings_str = "-" + opts.rgb
        variant= "rgb"

    if opts.version:
        version = opts.version
    else:
        version = 0
    exp_name = opts.game + "_seed" + str(opts.seed) + settings_str

    return exp_name, env_str, hide_properties, pruned_ff_name, variant, version


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
    parser.add_argument("-v", "--version", type=str, required=False, help="specify which trained version. standard selects highest number")
    parser.add_argument("--rgb", required= False, choices=["rgbv4", "rgbv5"], help="rgb observation space")
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

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
        variant =  "iscobots"
    if opts.prune == "default":
        settings_str += "_pruned-default"
    if opts.prune == "external":
        settings_str += "_pruned-external"
    if opts.exclude_properties:
        settings_str += '_excludeproperties'
        hide_properties = True

    #override setting str if rgb
    if opts.rgb:
        settings_str = "-" + opts.rgb
        variant= "rgb"
    exp_name = opts.game + "_seed" + str(opts. seed) + settings_str

    if opts.version:
        version = opts.version
    else:
        version = 0

    return exp_name, env_str, hide_properties, pruned_ff_name, opts.times, variant, version


def get_highest_version(agent):
    version = "-version"
    full_path = Path(agent)
    exp_name = full_path.name  # Extract the experiment name (e.g., 'something')
    base_path = full_path.parent  # Get the parent directory (e.g., 'pathtosomething')

    highest_version = 1
    version_pattern = re.compile(rf"{exp_name}-version(\d+)")  # Regex to match 'exp_name-versionX'

    for directory in base_path.iterdir():
        if directory.is_dir():  # Ensure it's a directory
            match = version_pattern.match(directory.name)
            if match:
                version_num = int(match.group(1))  # Extract the version number
                highest_version = max(highest_version, version_num)
    return version + str(highest_version)