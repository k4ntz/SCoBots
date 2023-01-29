import math
import numpy as np
# calculate running stats (welford's algo)
# https://www.johndcook.com/blog/standard_deviation/
#
class RunningStats:

    def __init__(self, n=0, m=0, s=0):
        self.n = n
        self.old_m = m
        self.old_s = s
        self.new_m = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


class Normalizer:

    def __init__(self, v_size, clip_value=0, stats=[]):
        self.running_stats = []            
        for _ in range(v_size):
            self.running_stats.append(RunningStats())
        self.n = len(self.running_stats)
        if len(stats) > 0:
            self._validate(v_size, stats)
            self.set_state(stats)
        self.clip_v = clip_value
 
    def set_state(self, stats):
        self._validate(self.n, stats)
        for i in range(self.n):
            self.running_stats[i].n = stats[i]["n"]
            self.running_stats[i].old_m = stats[i]["m"]
            self.running_stats[i].old_s = stats[i]["s"]

    def get_state(self):
        out = []
        for stat in self.running_stats:
            out.append({"n": stat.n, "m": stat.old_m, "s": stat.old_s})
        return out

    def normalize(self, vec):
        try:      
            out = []
            for i in range(self.n):
                self.running_stats[i].push(vec[i])
                mean = self.running_stats[i].mean()
                std = self.running_stats[i].standard_deviation()
                normed = (vec[i] - mean) / max(std, 1e-6)
                out.append(normed)
            if self.clip_v != 0:
                return np.clip(out, -self.clip_v, self.clip_v).tolist()
            return out
        except IndexError:
            print(f"normalizer dimension mismatch. excepts: {self.n}, got: {len(vec)}")
            exit()
    
    def _validate(self, expected_size, target):
        try:
            assert expected_size == len(target)
        except:
            print(f"normalizer dimension mismatch. excepts: {expected_size}, got: {len(target)}")
            exit()