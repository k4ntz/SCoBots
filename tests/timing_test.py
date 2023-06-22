from scobi import Environment
from ocatari.core import OCAtari
from typing import Tuple

import numpy as np
import gymnasium as gym
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
EPS = np.finfo(np.float32).eps.item()



# from code
def calc_lin_traj(a_position: Tuple[int, int], b_history: Tuple[int, int, int, int]) -> Tuple[int, int]:
    obj1 = a_position
    obj2 = b_history[0:2]
    obj2_past = b_history[2:4]
    x, y = obj2, obj2_past
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    disty = np.int16(m * obj1[0] + c) - obj1[1]
    distx = np.int16((obj1[1] - c) / (m+EPS))  - obj1[0]
    return distx, disty

# np polyfit
def polyfit(a_position: Tuple[int, int], b_history: Tuple[int, int, int, int]) -> Tuple[int, int]:
    obj2 = b_history[0:2]
    obj2_past = b_history[2:4]
    return np.polyfit([obj2[0], obj2_past[0]], [obj2[1], obj2_past[1]], 1)

# new
def coor_distance_to_traj(a_position: Tuple[int, int], b_history: Tuple[int, int, int, int]) -> Tuple[int, int]:
    m = (b_history[3] - b_history[1]) / (b_history[2] - b_history[0] + EPS)  # slope  m = (y2 - y1) / (x2 - x1)
    b = b_history[1] - m * b_history[0] # b = y - mx
    disty = (m * a_position[0] + b) - a_position[1] # delta_y = y_a - (m * x_a + b)
    distx = ((a_position[1] - b) / m) - a_position[0] # delta_x = x_a - (y_a - b) / m
    return distx, disty

# new abs distance
def abs_distance_to_traj(a_position: Tuple[int, int], b_history: Tuple[int, int, int, int]) -> Tuple[int, int]:
    m = (b_history[3] - b_history[1]) / (b_history[2] - b_history[0] + EPS)  # slope  m = (y2 - y1) / (x2 - x1)
    b = b_history[1] - m * b_history[0] # b = y - mx
    r = np.abs(-m * a_position[0]  + a_position[1] - b)  / np.sqrt(np.power(-m, 2) + np.power(1, 2)) 
    return r



# 10000 loops, best of 5: 33.3 usec per loop
def orig():
    a = np.random.random_sample((2,))
    b = np.random.random_sample((4,))
    calc_lin_traj(a, b)

# 10000 loops, best of 5: 44.9 usec per loop
def poly():
    a = np.random.random_sample((2,))
    b = np.random.random_sample((4,))
    polyfit(a, b)

# 10000 loops, best of 5: 1.93 usec per loop
def fast_dist():
    a = np.random.random_sample((2,))
    b = np.random.random_sample((4,))
    coor_distance_to_traj(a, b)

# 10000 loops, best of 5: 4.72 usec per loop
def fast_abs_dist():
    a = np.random.random_sample((2,))
    b = np.random.random_sample((4,))
    abs_distance_to_traj(a, b)



env_str='PongDeterministic-v4'
raw_env = gym.make(env_str)
oc_env =  OCAtari(env_name=env_str, mode="revised")
scobi_env = Environment(env_name=env_str)
scobi_pruned_env =  Environment(env_name='BowlingDeterministic-v4', interactive=True, focus_dir="experiments/my_focusfiles", focus_file="pruned_bowling.yaml")

raw_env.reset()
oc_env.reset()
scobi_env.reset()
scobi_pruned_env.reset()

# 10000 loops, best of 5: 41.6 usec per loop
def raw_step():
    raw_env.step(0)

# 10000 loops, best of 5: 171 usec per loop
def oc_step():
    oc_env.step(0)

# 10000 loops, best of 5: 692 usec per loop                             new: 10000 loops, best of 5: 340 usec per loop
def scobi_step():
    scobi_env.step(0)


# base ocatari:         10000 loops, best of 5: 188 usec per loop       new: unchanged
# ret oca, no driver:   10000 loops, best of 5: 191 usec per loop       new: unchanged
# driver, no concepts:  10000 loops, best of 5: 196 usec per loop       new: unchanged
# -----------------------------------------------------------
# only color:           10000 loops, best of 5: 238 usec per loop       new: unchanged
# only euclid:          10000 loops, best of 5: 249 usec per loop       new: unchanged
# all - lin_traj        10000 loops, best of 5: 303 usec per loop       new: unchanged
# only lin_traj:        10000 loops, best of 5: 620 usec per loop       new: 10000 loops, best of 5: 267 usec per loop
# all                   10000 loops, best of 5: 692 usec per loop       new: 10000 loops, best of 5: 340 usec per loop
def scobi_pruned_step():
    scobi_pruned_env.step(0)

#import cProfile, pstats
#profiler = cProfile.Profile()
#profiler.enable()
#for _ in range(10000):
#    scobi_step()
#profiler.disable()
#stats = pstats.Stats(profiler).sort_stats('tottime')
#stats.print_stats()
#exit()