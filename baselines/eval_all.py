import numpy as np
from scobi import Environment
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import  WarpFrame, AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from multiprocessing import JoinableQueue
from multiprocessing import Process, Value


def main():
    envs =["Seaquest", "Kangaroo", "Asterix", "Bowling", "Tennis", "Boxing", "Freeway", "Skiing", "Pong"] 
    check_dir = "baselines_checkpoints"
    variants = ["rgbv5"] #["scobots"] #["scobots", "iscobots"]#, "rgb"]
    eval_env_seeds = [123, 456, 789, 1011] # [84, 58*2, 74*2]  #[123, 456, 789, 1011]
    episodes_per_seed = 5
    checkpoint_str = "best_model" #"model_5000000_steps"
    vecnorm_str = "best_vecnormalize.pkl"
    eval_results_pkl_path = Path("rgb-v5_eval_results.pkl")
    eval_results_csv_path = Path("rgb-v5_eval_results.csv")
    results_header = ["env", "variant", "train_seed", "eval_seed", "episodes", "reward_mean", "reward_std", "steps_mean", "steps_std"]
    EVALUATORS = 4

    def run_list():
        task = {"env": "", "variant" : "", "model": [], "eval_seed": ""}
        for e in envs:
            task["env"] = e
            for v in variants:
                task["variant"] = v
                p = Path(check_dir, v)
                for path in Path(p).iterdir():
                    if path.is_dir() and e in str(path):
                        task["model"] = path
                        for s in eval_env_seeds:
                            task["eval_seed"] = s
                            yield task


    def evaluate(jobq : JoinableQueue, doneq : JoinableQueue):
        while True:
            if jobq.empty():
                break 
            task = jobq.get()
            env_str = task["env"]
            model_dir = task["model"]
            eval_seed = task["eval_seed"]
            variant = task["variant"]
            vecnorm_path = Path(model_dir,  vecnorm_str)
            model_path = Path(model_dir, checkpoint_str)
            train_seed = model_dir.name.split("_")[1][1:]
            trainseedsplit = train_seed.split("-")
            if len(trainseedsplit) > 1:
                train_seed = trainseedsplit[0]
            pruned_ff_name = None
            if task["variant"] == "iscobots":
                pruned_ff_name = f"pruned_{env_str.lower()}.yaml"
            atari_env_str = "ALE/" + env_str +"-v5"
            
            if variant == "rgbv4":
                atari_env_str = env_str + "NoFrameskip-v4"
                eval_wrapper_params = {"noop_max" : 30, "frame_skip" : 4, "screen_size": 84, "terminal_on_life_loss": False, "clip_reward" : False, "action_repeat_probability" : 0.0} # remaining values are part of AtariWrapper
                env = make_vec_env(atari_env_str, seed=eval_seed, wrapper_class=AtariWrapper, wrapper_kwargs=eval_wrapper_params)
                env = VecTransposeImage(env)
            elif variant == "rgbv5":
                eval_wrapper_params = {"noop_max" : 0, "frame_skip" : 1, "screen_size": 84, "terminal_on_life_loss": False, "clip_reward" : False, "action_repeat_probability" : 0.0} # remaining values are part of AtariWrapper
                env = make_vec_env(atari_env_str, seed=eval_seed, wrapper_class=AtariWrapper, wrapper_kwargs=eval_wrapper_params)
                env = VecTransposeImage(env)
            else:
                env = Environment(atari_env_str, focus_file=pruned_ff_name, silent=True, refresh_yaml=False)
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
                if current_episode == episodes_per_seed:
                    #["env", "variant", "train_seed", "eval_seed", "episodes", "reward_mean", "reward_std", "steps_mean", "steaps_std"]
                    result_record = [env_str, variant, train_seed, eval_seed, episodes_per_seed, np.mean(episode_rewards), np.std(episode_rewards), np.mean(steps), np.std(steps)]
                    doneq.put(result_record)
                    jobq.task_done()
                    break

        
    
    def flush(doneq : JoinableQueue, pbar):
        while True:
            record = doneq.get()
            if eval_results_pkl_path.exists():
                results_df = pd.read_pickle(eval_results_pkl_path)
                results_df.loc[len(results_df)] = record
            else:
                results_df = pd.DataFrame([record], columns=results_header)
            results_df.to_pickle(eval_results_pkl_path)
            results_df.to_csv(eval_results_csv_path)
            pbar.update(episodes_per_seed)
            doneq.task_done()



    for e in envs: #refresh default focus files
        atari_env_str = "ALE/" + e +"-v5"
        Environment(atari_env_str, silent=True, refresh_yaml=True)
    
    pbarsize = 0
    for r in run_list():
        pbarsize += 1
    pbarsize *= episodes_per_seed
    pbar = tqdm(total=pbarsize)


    jobq = JoinableQueue()
    doneq = JoinableQueue()
    rlist = run_list()
    for run in rlist:
        job = run.copy()
        jobq.put(job)


    workers = []
    for _ in range(EVALUATORS):
        t = Process(target=evaluate, args=(jobq, doneq))
        t.daemon = True
        workers.append(t)
    for w in workers:
        w.start()


    
    flusher = Process(target=flush, args=(doneq, pbar))
    flusher.daemon = True
    flusher.start()

    jobq.join()
    for w in workers:
        w.join()
    doneq.join()

 
if __name__ == '__main__':
    main()