from scobi import Environment
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning) 
from scobi.utils import logging
logging.SILENT = False

# Minimal init, not interactive, no focus dir/file specified
os.environ["SCOBI_OBJ_EXTRACTOR"] = "Noisy_OC_Atari"
env = Environment(env_name='ALE/Pong-v5')
env.reset()
obs, reward, truncated, terminated, info = env.step(1)
for i in range(10):
    env.step(1)
env.close()
aspace = env.action_space
aspace_desc = env.action_space_description
ospace = env.observation_space
ospace_desc = env.observation_space_description
original_reward = env.original_reward
scobi_reward = reward
original_obs = env.original_obs

# print(aspace)       # Discrete(6)
# print(aspace_desc)  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
# print(ospace)       # Box(-1000.0, 1000.0, (81,), float32)
# print(ospace_desc)  # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
# print(obs)          # [0.0, 0.0, 16.0, 1.0, 140.0, ...

exit()
# Extensive init, interactive, custom fcous dir and focus file
env = Environment(env_name='SkiingDeterministic-v4', interactive=True, focus_dir="experiments/my_focusfiles", focus_file="pruned_skiing.yaml")
env.reset()
obs, reward, truncated, terminated, info = env.step(1)
env.close()
aspace = env.action_space
aspace_desc = env.action_space_description
ospace = env.observation_space
ospace_desc = env.observation_space_description

print(aspace)       # Discrete(4)
print(aspace_desc)  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
print(ospace)       # Box(-1000.0, 1000.0, (9,), float32)
print(ospace_desc)  # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
print(obs)          # [0.0, 0.0, 16.0, 1.0, 140.0, ...