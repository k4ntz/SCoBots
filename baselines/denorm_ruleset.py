from experiments.ppo_utils import load_ppo_env

REMIX=False
if not REMIX:
    import os.path as osp
    from scobi import Environment
    import math
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3 import PPO
    import json

    pruned_ff_name = "default_focus_Pong-v4.yaml"
    focus_dir = "focusfiles"
    ckpt_path = "baselines_checkpoints"
    exp_name = "Pong_s42_re-v3_gtdata"
    env = load_ppo_env(
        focus_dir=focus_dir,
        ff_name=pruned_ff_name,
        exp_name=exp_name,
        ckpt_path=ckpt_path,
        as_vecenv=True
    )

    fnames = env.envs[0].get_vector_entry_descriptions()
    fnames = [f.replace(" ", "") for f in fnames]
    mean = env.obs_rms.mean
    variance = env.obs_rms.var

    # create and save denorm dict
    denorm_dict = {}
    for mean, variance, name in zip(mean, variance, fnames):
        standard_deviation = math.sqrt(variance)
        denorm_dict[name] = (mean, standard_deviation)
    with open("denorm_dict.json", "w") as f:
        json.dump(denorm_dict, f)

else:
    from remix.rules.ruleset import Ruleset
    import json
    with open("denorm_dict.json", "r") as f:
        denorm_dict = json.load(f)
    ruleset_filepath = "output.rules"
    ruleset = Ruleset().from_file(ruleset_filepath)
    
    for rule in ruleset:
        for p in rule.premise:
            terms_set = p.terms
            for term in terms_set:
                m, s = denorm_dict[term.variable]
                denorm_value = int((term.threshold * s) + m)
                term.threshold = denorm_value
    
    ruleset.to_file("denormed.rules")


