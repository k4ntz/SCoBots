from stable_baselines3 import PPO
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tqdm import tqdm
from joblib import dump
from statistics import mean
from copy import deepcopy
from operator import itemgetter
from gymnasium import Env
from pathlib import Path
import os
import torch
import time

# from viperoc repository
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


class DecisionTreeExtractor: #Dagger
    def __init__(self, model: PPO, dtpolicy: DecisionTreeClassifier, env: Env, data_per_iter: int=20_000):
        self.model = model
        self.env = env # is vectorized
        self.data_per_iter = data_per_iter
        self.dt = dtpolicy

    def collect_data(self):
        S, A = [], []
        s = self.env.reset()
        for i in tqdm(range(self.data_per_iter)):
            action = self.model.predict(s, deterministic=True)[0]
            S.append(s[0]) #unvec
            A.append(action[0]) #unvec
            s, _, done, _ = self.env.step(action)
            if done:
                s = self.env.reset()
        return np.array(S), np.array(A)

    def collect_data_dt(self,):
        S = []
        episodes = []
        ep_reward = 0
        s = self.env.reset()
        for i in range(self.data_per_iter):
            action = self.dt.predict(s.reshape(1, -1))
            S.append(s[0]) #unvec
            s, r, done, infos = self.env.step(action)
            ep_reward += r
            if done:
                s = self.env.reset()
                episodes.append(ep_reward)
                ep_reward = 0
        if len(episodes) < 1:
            episodes.append(ep_reward)
        return S, np.mean(episodes)

    def fit_DT(self, S, A):
        ## sampling
        self.dt.fit(S, A)
        acc = self.dt.score(S, A)
        return acc

    def imitate(self, nb_iter: int):
        start_time = time.time()
        self.list_acc, self.list_eval, self.list_dt, self.times = [], [], [], []
        DS, DA = self.collect_data()
        acc_dt = self.fit_DT(DS, DA)
        S_dt, eval_dt = self.collect_data_dt()
        self.times.append(time.time()-start_time)

        print("Accuracy: {} - Evaluation: {}".format(acc_dt, eval_dt))
        self.list_dt.append(deepcopy(self.dt))
        self.list_acc.append(acc_dt)
        self.list_eval.append(eval_dt)
        DS = np.concatenate((DS, S_dt))
        DA = np.concatenate((DA, self.model.predict(S_dt)[0]))
        
        for _ in range(nb_iter - 1):
            acc_dt = self.fit_DT(DS, DA)
            S_dt, eval_dt = self.collect_data_dt()
            self.times.append(time.time()-start_time)

            print("Accuracy: {} - Evaluation: {}".format(acc_dt, eval_dt))
            self.list_dt.append(deepcopy(self.dt))
            self.list_acc.append(acc_dt)
            self.list_eval.append(eval_dt)
            DS = np.concatenate((DS, S_dt))
            DA = np.concatenate((DA, self.model.predict(S_dt)[0]))

    def save_best_tree(self, out_path):
        trees_path = out_path / Path("viper_trees")
        trees_path.mkdir(parents=True, exist_ok=True)

        for j, tree in enumerate(self.list_dt):
            fpath = "Tree-%s_%s.viper" % (j, self.list_eval[j])
            dump(tree, trees_path / fpath)

        index, element = max(enumerate(self.list_eval), key=itemgetter(1))
        self.best_dt = self.list_dt[index]
        best_fpath = "Tree-"+str(element) + "_best.viper"
        dump(self.best_dt, out_path / best_fpath)
    

class VIPER(DecisionTreeExtractor):
    def __init__(self, model: PPO, dtpolicy: DecisionTreeClassifier, env: Env, rtpt, data_per_iter: int=20_000):
        super().__init__(model, dtpolicy, env, data_per_iter)
        self.Q = LogProbQ(self.model, self.env)
        self.rtpt = rtpt

    def fit_DT(self, S, A, weights):
        self.dt.fit(S, A, weights)
        acc = self.dt.score(S, A, weights)
        return acc
    
    def imitate(self, nb_iter: int):
        start_time = time.time()
        self.list_acc, self.list_eval, self.list_dt, self.times =[], [], [], []
        DS, DA = self.collect_data()
        weights = [self.Q.get_disagreement_cost(s).item() for s in DS] # could be sped up

        acc_dt = self.fit_DT(DS, DA, weights)
        S_dt, eval_dt = self.collect_data_dt()
        self.times.append(time.time()-start_time)
        print("Accuracy: {} - Evaluation: {}".format(acc_dt, eval_dt))
        self.list_dt.append(deepcopy(self.dt))
        self.list_acc.append(acc_dt)
        self.list_eval.append(eval_dt)
        DS = np.concatenate((DS, S_dt))
        DA = np.concatenate((DA, self.model.predict(S_dt)[0]))
        weights += [self.Q.get_disagreement_cost(s).item() for s in S_dt]
        
        for _ in range(nb_iter - 1):
            self.rtpt.step()
            acc_dt = self.fit_DT(DS, DA, weights)
            S_dt, eval_dt = self.collect_data_dt()
            self.times.append(time.time()-start_time)
            print("Accuracy: {} - Evaluation: {}".format(acc_dt, eval_dt))
            self.list_dt.append(deepcopy(self.dt))
            self.list_acc.append(acc_dt)
            self.list_eval.append(eval_dt)
            DS = np.concatenate((DS, S_dt))
            DA = np.concatenate((DA, self.model.predict(S_dt)[0]))
            weights += [self.Q.get_disagreement_cost(s).item() for s in S_dt]
