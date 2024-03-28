# scobi
Successive Concepet Bottleneck Interface

you need ocatari

## How to use (example)
Select object extractor by setting env var 'SCOBI_OBJ_EXTRACTOR'. Supported values: 'OC_Atari', 'Noisy_OC_Atari'.


```python
from scobi import Environment

# Minimal init, not interactive, no focus dir/file specified, empirical observation space normalization active
env = Environment(env_name='PongDeterministic-v4')
env.reset()
obs, scobi_reward, truncated, terminated, info = env.step(1)
env.action_space                    # Discrete(6)
env.action_space_description        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
env.observation_space               # Box(-1000.0, 1000.0, (81,), float32)
env.observation_space_description   # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
env.original_reward                 # ALE reward
env.original_obs                    # ALE obs
env.close()

# Extensive init, interactive, custom fcous dir and focus file, empirical observation space normalization not active
env = Environment(env_name='PongDeterministic-v4', focus_dir="my_focusfiles", focus_file="pruned_pong.yaml")
env.reset()
obs, scobi_reward, truncated, terminated, info = env.step(1)
env.action_space                    # Discrete(4)
env.action_space_description        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
env.observation_space               # Box(-1000.0, 1000.0, (12,), float32)
env.observation_space_description   # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
env.original_reward                 # ALE reward
env.original_obs                    # ALE obs
env.close()
```

## Where to start

Reinforce Training: ```/experiments```
```bash
python train.py --config configs/scobi/re-pong.yaml seed 42
```

PPO Training ```/baselines```
```bash
python train.py -g Pong -s 42 -c 8 -r env
```
