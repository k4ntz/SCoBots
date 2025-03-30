import time
from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import numpy as np
import torch
from gymnasium import Env
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# train VIPER tree using denormalized data
cuda = torch.device("cuda")

class LogProbQ:
    def __init__(self, stochastic_pol: PPO, env: Env):
        self.pol = stochastic_pol
        self.env = env

    def q(self, s):
        s = torch.Tensor(s).to(cuda)
        s_repeat = s.repeat(self.env.action_space.n,1)
        with torch.no_grad():
            _, s_a_log_probs, _ = self.pol.policy.evaluate_actions(s_repeat, torch.arange(self.env.action_space.n).reshape(-1, 1).to(cuda))
        return s_a_log_probs

    def get_disagreement_cost(self, s):
        log_prob = self.q(s)
        return log_prob.mean() - log_prob.min()


class DenormalizedVIPER:
    def __init__(self, model: PPO, dtpolicy: DecisionTreeClassifier, env: Env, data_per_iter: int=30_000, rtpt=None):
        """initialize DenormalizedVIPER
        
        Args:
            model: trained model
            dtpolicy: decision tree classifier
            env: environment (must be vectorized and normalized)
            data_per_iter: number of data collected per iteration
            rtpt: RTPT instance for progress tracking
        """
        self.model = model
        self.env = env  # must be VecNormalize environment
        self.data_per_iter = data_per_iter
        self.dt = dtpolicy
        self.Q = LogProbQ(self.model, self.env)
        self.rtpt = rtpt
        
        # ensure the environment is VecNormalize type
        if not isinstance(env, VecNormalize):
            raise ValueError("environment must be VecNormalize type to get denormalized data")
            
        # set the parameters of the normalized environment
        if isinstance(env, VecNormalize):
            env.training = False  # stop updating the running mean/variance
            env.norm_reward = False  # do not normalize the reward

    def _get_original_obs(self, obs):
        """get the original unnormalized observation data"""
        if isinstance(self.env, VecNormalize):
            # VecNormalize provides the get_original_obs method
            return self.env.get_original_obs()
        return obs

    def collect_data(self):
        """collect training data, use denormalized data"""
        S, A, S_orig = [], [], []  # save both normalized and original data
        obs = self.env.reset()
        s_orig = self._get_original_obs(obs)
        
        for i in tqdm(range(self.data_per_iter)):
            action, _ = self.model.predict(obs, deterministic=True)
            S.append(obs[0])  # save normalized data (for getting actions)
            S_orig.append(s_orig[0])  # save original data (for training the tree)
            A.append(action[0])
            obs, _, done, info = self.env.step(action)
            s_orig = self._get_original_obs(obs)
            if done:
                obs = self.env.reset()
                s_orig = self._get_original_obs(obs)
                
        return np.array(S), np.array(A), np.array(S_orig)

    def collect_data_dt(self):
        """use the decision tree to collect data to evaluate the performance"""
        S_orig = []
        episodes = []
        ep_reward = 0
        obs = self.env.reset()
        s_orig = self._get_original_obs(obs)
        
        for i in range(self.data_per_iter):
            # use the original (non-normalized) data when predicting
            action = self.dt.predict(s_orig.reshape(1, -1))
                
            S_orig.append(s_orig[0])
            obs, r, done, info = self.env.step(action)
            s_orig = self._get_original_obs(obs)
            ep_reward += r
            if done:
                obs = self.env.reset()
                s_orig = self._get_original_obs(obs)
                episodes.append(ep_reward)
                ep_reward = 0
                
        if len(episodes) < 1:
            episodes.append(ep_reward)
        return S_orig, np.mean(episodes)

    def fit_DT(self, S, A, weights):
        """train the decision tree (with weights)"""
        self.dt.fit(S, A, weights)
        acc = self.dt.score(S, A, weights)
        return acc

    def imitate(self, nb_iter: int):
        """imitation learning process, use denormalized data and weights to train the tree"""
        start_time = time.time()
        self.list_acc, self.list_eval, self.list_dt, self.times = [], [], [], []
        
        # initial data collection
        DS, DA, DS_orig = self.collect_data()
        weights = [self.Q.get_disagreement_cost(s).item() for s in DS]  # use normalized data to calculate the weights
        
        # train the decision tree using the original data
        acc_dt = self.fit_DT(DS_orig, DA, weights)
        
        # evaluate the performance
        S_dt, eval_dt = self.collect_data_dt()
        self.times.append(time.time()-start_time)
        
        print(f"using denormalized data - accuracy: {acc_dt} - evaluation: {eval_dt}")
        self.list_dt.append(deepcopy(self.dt))
        self.list_acc.append(acc_dt)
        self.list_eval.append(eval_dt)
        
        # subsequent iterations
        for _ in range(nb_iter - 1):
            if self.rtpt:
                self.rtpt.step()
                
            # collect more data
            new_DS, new_DA, new_DS_orig = self.collect_data()
            
            # merge data
            DS = np.concatenate((DS, new_DS))
            DA = np.concatenate((DA, new_DA))
            DS_orig = np.concatenate((DS_orig, new_DS_orig))
            
            # calculate the weights of the new data
            new_weights = [self.Q.get_disagreement_cost(s).item() for s in new_DS]
            weights = weights + new_weights
            
            # train the decision tree
            acc_dt = self.fit_DT(DS_orig, DA, weights)
            
            # evaluate the performance
            S_dt, eval_dt = self.collect_data_dt()
            self.times.append(time.time()-start_time)
            
            print(f"using denormalized data - accuracy: {acc_dt} - evaluation: {eval_dt}")
            self.list_dt.append(deepcopy(self.dt))
            self.list_acc.append(acc_dt)
            self.list_eval.append(eval_dt)
    
    def save_best_tree(self, out_path):
        """save the best decision tree"""
        trees_path = out_path / Path("viper_trees")
        trees_path.mkdir(parents=True, exist_ok=True)

        # save all trees
        for j, tree in enumerate(self.list_dt):
            fpath = f"Tree-{j}_{self.list_eval[j]}_denorm.viper"
            dump(tree, trees_path / fpath)

        # find and save the best tree
        index, element = max(enumerate(self.list_eval), key=itemgetter(1))
        self.best_dt = self.list_dt[index]
        best_fpath = f"Tree-{element}_denorm_best.viper"
        dump(self.best_dt, out_path / best_fpath)
        
        # save the training information
        info_path = out_path / f"viper_training_info_denorm.txt"
        with open(info_path, "w") as f:
            f.write(f"using denormalized data training\n")
            f.write(f"best tree reward: {element}\n")
            f.write(f"training iterations: {len(self.list_eval)}\n")
            f.write(f"training time: {self.times[-1]:.2f} seconds\n")
            f.write("\naccuracy history:\n")
            for i, acc in enumerate(self.list_acc):
                f.write(f"iteration {i}: {acc}\n")
            
            f.write("\nevaluation reward history:\n")
            for i, reward in enumerate(self.list_eval):
                f.write(f"iteration {i}: {reward}\n")