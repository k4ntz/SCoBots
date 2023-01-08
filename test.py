from scobi import Environment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Minimal init, not interactive, no focus dir/file specified
env = Environment(env_name='PongDeterministic-v4')
env.reset()
obs, reward, truncated, terminated, info = env.step(1)
env.close()
aspace = env.action_space
aspace_desc = env.action_space_description
ospace = env.observation_space
ospace_desc = env.observation_space_description


# Extensive init, interactive, custom fcous dir and focus file
env = Environment(env_name='PongDeterministic-v4', interactive=True, focus_dir="experiments/my_focusfiles", focus_file="2_actions_pong.yaml")
env.reset()
obs, reward, truncated, terminated, info = env.step(1)
env.close()
aspace = env.action_space
aspace_desc = env.action_space_description
ospace = env.observation_space
ospace_desc = env.observation_space_description






#print(aspace)       # Discrete(4)
#print(aspace_desc)  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
#print(ospace)       # Box(-1000.0, 1000.0, (81,), float32)
#print(ospace_desc)  # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
#print(obs)          # [0.0, 0.0, 16.0, 1.0, 140.0, ...