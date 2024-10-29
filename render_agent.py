from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from scobi import Environment
from utils.parser.parser import render_parser, convert_args, get_highest_version
from utils.renderer import Renderer
from viper_extract import DTClassifierModel
from joblib import load


def flist(l):
    return ["%.2f" % e for e in l]

def _load_viper(exp_name, path_provided):
    if path_provided:
        viper_path = Path(exp_name)
        model = load(sorted(viper_path.glob("*_best.viper"))[0])
    else:
        viper_path = Path("resources/viper_extracts/extract_output", exp_name + "-extraction")
        model = load(sorted(viper_path.glob("*_best.viper"))[0])

    wrapped = DTClassifierModel(model)

    return wrapped


def main():
    opts = render_parser()
    exp_name, env_str, hide_properties, pruned_ff_name, variant, version, viper = convert_args(opts)
    
    if version == 0:
        version = get_highest_version(exp_name)

    exp_name += version

    checkpoint_str = "best_model" # "model_5000000_steps" #"best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    model_path = Path("resources/checkpoints", exp_name, checkpoint_str)
    vecnorm_path = Path("resources/checkpoints",  exp_name, vecnorm_str)
    EVAL_ENV_SEED = 84

    if variant == "rgb":
        env = make_vec_env(env_str, seed=EVAL_ENV_SEED, wrapper_class=WarpFrame)
    else:
        env = Environment(env_str,
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
    else:
        model = PPO.load(model_path)
    obs = env.reset()
    renderer = Renderer(env, model, opts.record, opts.nb_frames)
    renderer.print_reward = opts.print_reward
    renderer.run()

if __name__ == '__main__':
    main()