from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from scobi import Environment
from utils.parser.parser import render_parser, get_highest_version
from utils.renderer import Renderer
from viper_extract import DTClassifierModel
from joblib import load


def flist(l):
    return ["%.2f" % e for e in l]

# Helper function to load from a dt and not a checkpoint directly
def _load_viper(exp_name, path_provided):
    if path_provided:
        viper_path = Path(exp_name)
        model = load(sorted(viper_path.glob("*_best.viper"))[0])
    else:
        viper_path = Path("resources/viper_extracts/extract_output", exp_name + "-extraction")
        model = load(sorted(viper_path.glob("*_best.viper"))[0])

    wrapped = DTClassifierModel(model)

    return wrapped

def _load_interpreter(exp_name, path_provided):
    """Helper function to load from a interpreter tree"""
    if path_provided:
        tree_path = Path(exp_name)
        tree = load(sorted(tree_path.glob("tree.interpreter"))[0])
    else:
        tree_path = Path("resources/interpreter_extract/extract_output", exp_name + "-extraction")
        tree = load(sorted(tree_path.glob("tree.interpreter"))[0])
        print("Loading interpreter tree from " + str(tree_path))
        print(tree) 

    return tree

# Helper function ensuring that a checkpoint has completed training
def _ensure_completeness(path):
    checkpoint = path / "best_model.zip"
    return checkpoint.is_file()


def main():
    flag_dictionary = render_parser()
    version = int(flag_dictionary["version"])
    exp_name = flag_dictionary["exp_name"]
    variant = flag_dictionary["variant"]
    env_str = flag_dictionary["env_str"]
    pruned_ff_name = flag_dictionary["pruned_ff_name"]
    hide_properties = flag_dictionary["hide_properties"]
    viper = flag_dictionary["viper"]
    interpreter = flag_dictionary["interpreter"]
    record = flag_dictionary["record"]
    nb_frames = flag_dictionary["nb_frames"]
    print_reward = flag_dictionary["print_reward"]
    
    if version == -1:
        version = get_highest_version(exp_name)
    elif version == 0:
        version = ""

    exp_name += str(version)

    checkpoint_str = "best_model" # "model_5000000_steps" #"best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("resources/checkpoints", exp_name, checkpoint_str)
    vecnorm_path = Path("resources/checkpoints",  exp_name, vecnorm_str)
    ff_file_path = Path("resources/checkpoints", exp_name)
    EVAL_ENV_SEED = 84
    if not _ensure_completeness(ff_file_path):
        print("The folder " + str(ff_file_path) + " does not contain a completed training checkpoint.")
        return
    if variant == "rgb":
        env = make_vec_env(env_str, seed=EVAL_ENV_SEED, wrapper_class=WarpFrame)
    else:
        env = Environment(env_str,
                          focus_dir=ff_file_path,
                          focus_file=pruned_ff_name,
                          hide_properties=hide_properties,
                          draw_features=True, # implement feature attribution
                          reward=0) #env reward only for evaluation

        _, _ = env.reset(seed=EVAL_ENV_SEED)
        dummy_vecenv = DummyVecEnv([lambda :  env])
        env = VecNormalize.load(vecnorm_path, dummy_vecenv)
        env.training = False
        env.norm_reward = False
    if viper:
        print("loading viper tree of " + exp_name)
        if isinstance(viper, str):
            model = _load_viper(viper, True)
        else:
            model = _load_viper(exp_name, False)
    elif interpreter: 
        print("loading interpreter tree of " + exp_name)
        if isinstance(interpreter, str):
            model = _load_interpreter(interpreter, True)
        else:
            model = _load_interpreter(exp_name, False)
    else:
        model = PPO.load(model_path)
    obs = env.reset()
    renderer = Renderer(env, model, ff_file_path, record, nb_frames)
    renderer.print_reward = print_reward
    renderer.run()

if __name__ == '__main__':
    main()