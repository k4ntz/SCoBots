from typing import Tuple
import os.path as osp
import json
import numpy as np

epsilon = 1e-4
clip_obs = 10.0

class MyNormalizer:

    def __init__(self, mean, var, clip_obs, epsilon):
        """
        Normalizes the observation to have zero mean and unit variance.
        :param obs_shape: observation shape
        :param clip_obs: clip observations to avoid numerical instabilities
        :param epsilon: helps with arithmetic issues
        """
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.mean = mean
        self.var = var

    #def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
    #    """
    #    Helper to normalize observation.
    #    :param obs:
    #    :return: normalized observation
    #    """
    #    return np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)
    
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :return: normalized observation
        """
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

def save_normalizer(vec_normalizer, path):
    #numpy arrays
    mean = vec_normalizer.obs_rms.mean.tolist()
    variance = vec_normalizer.obs_rms.var.tolist()

    #scalar values
    epsilon = vec_normalizer.epsilon
    clip_obs = vec_normalizer.clip_obs

    #save as json
    data = {"mean": mean, "variance": variance, "epsilon": epsilon, "clip_obs": clip_obs}
    with open(path, "w") as f:
        json.dump(data, f)

def load_normalizer(path):
    with open(path, "r") as f:
        data = json.load(f)
    mean = np.array(data["mean"])
    variance = np.array(data["variance"])
    epsilon = data["epsilon"]
    clip_obs = data["clip_obs"]
    normalizer = MyNormalizer(mean, variance, clip_obs, epsilon)
    return normalizer