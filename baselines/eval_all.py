import numpy as np
from scobi import Environment
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def main():
    envs = ["Bowling", "Skiing", "Pong", "Boxing", "Freeway", "Skiing"]
    check_dir = "baselines_checkpoints"
    variants = ["scobots", "iscobots"]#, "rgb"]
    eval_env_seeds = [123, 456, 789, 1011]
    episodes_per_seed = 10
    checkpoint_str = "best_model" #"best_model"
    vecnorm_str = "best_vecnormalize.pkl"
    run_results = []
    detailed_results_header = ["env", "variant", "train_seed", "eval_seed", "episodes", "mean", "std"]
    detailed_results_data = []

    def run_list():
        run_queue = {"env": "", "variant" : "", "models": []}
        for e in envs:
            run_queue["env"] = e
            for v in variants:
                run_list = []
                run_queue["variant"] = v
                p = Path(check_dir, v)
                for path in Path(p).iterdir():
                    if path.is_dir() and e in str(path):
                        run_list.append(path)
                        run_queue["models"] = run_list
                yield run_queue




    pbarsize = 0
    for a in run_list():
        pbarsize += len(a["models"])
    pbarsize = pbarsize * len(eval_env_seeds) * episodes_per_seed
    pbar = tqdm(total=pbarsize)
    rlist = run_list()
    for run in rlist:
        model_dir_results = []
        for model_dir in run["models"]:
            train_seed = model_dir.name.split("_")[1][1:]
            eval_seed_rewards = []
            env_str = run["env"]
            game_id = env_str.lower()
            pruned_ff_name = None
            if run["variant"] == "iscobots":
                pruned_ff_name = f"pruned_{game_id}.yaml"
            for eval_seed in eval_env_seeds:
                pbar.set_description(f"{env_str} {run['variant']} {eval_seed}")
                vecnorm_path = Path(model_dir,  vecnorm_str)
                model_path = Path(model_dir, checkpoint_str)
                
                atari_env_str = "ALE/" + env_str +"-v5"
                env = Environment(atari_env_str, focus_file=pruned_ff_name, silent=True)
                _, _ = env.reset(seed=eval_seed)
                dummy_vecenv = DummyVecEnv([lambda :  env])
                env = VecNormalize.load(vecnorm_path, dummy_vecenv)
                env.training = False
                env.norm_reward = False
                model = PPO.load(model_path)

                current_episode = 0
                episode_rewards = []
                steps = []
                current_rew = 0
                current_step = 0
                obs = env.reset()
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    current_rew += reward
                    current_step += 1
                    if done:
                        current_episode += 1
                        episode_rewards.append(current_rew)
                        steps.append(current_step)
                        current_rew = 0
                        current_step = 0
                        obs = env.reset()
                        pbar.update(1)
                    if current_episode == episodes_per_seed:
                        #print(episode_rewards)
                        #print(f"> mean: {np.mean(episode_rewards):.2f} mean_steps: {np.mean(steps):.2f}")
                        eval_seed_rewards.append(np.mean(episode_rewards))
                        break
            #print(eval_seed_rewards)
            #print(f"> mean: {np.mean(eval_seed_rewards):.2f}")
            model_dir_results.append(np.mean(eval_seed_rewards))
            #["env", "variant", "train_seed", "eval_seed", "episodes", "mean", "std"]
            result_record = [env_str, run["variant"], train_seed, eval_seed, episodes_per_seed, np.mean(eval_seed_rewards), np.std(eval_seed_rewards)]
            detailed_results_data.append(result_record)
       # print(model_dir_results)
       # print(f"> mean: {np.mean(model_dir_results):.2f}")
        run_results.append([env_str, run["variant"], np.mean(model_dir_results), np.std(model_dir_results)])
    detailed_results_df = pd.DataFrame(detailed_results_data, columns=detailed_results_header)
    print(detailed_results_df)
    agg_results_df = pd.DataFrame(run_results, columns=["env", "variant", "mean", "std"])
    print(agg_results_df)
    agg_results_df.to_pickle("eval_agg_results.pkl")
    detailed_results_df.to_pickle("eval_detailed_results.pkl")

if __name__ == '__main__':
    main()