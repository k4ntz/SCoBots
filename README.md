# scobi
Successive Concepet Bottleneck Interface

you need ocatari

raw -> object bottleneck -> concept bottleneck -> env

## How to use (example)

```python
from scobi import Environment

# Minimal init, not interactive, no focus dir/file specified, empirical observation space normalization active
env = Environment(env_name='PongDeterministic-v4')
env.reset()
obs, env_reward, scobi_reward, truncated, terminated, info, obs_raw = env.step(1)
env.action_space                    # Discrete(6)
env.action_space_description        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
env.observation_space               # Box(-1000.0, 1000.0, (81,), float32)
env.observation_space_description   # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
env.close()

# Extensive init, interactive, custom fcous dir and focus file, empirical observation space normalization not active
env = Environment(env_name='PongDeterministic-v4', interactive=True, focus_dir="experiments/my_focusfiles", focus_file="pruned_pong.yaml", obs_normalized=False)
env.reset()
obs, env_reward, scobi_reward, truncated, terminated, info, obs_raw = env.step(1)
env.action_space                    # Discrete(4)
env.action_space_description        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
env.observation_space               # Box(-1000.0, 1000.0, (12,), float32)
env.observation_space_description   # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
env.close()
```

## TODO
### Major
- unify tensoboard logging across algos (+ finegrained logging?)
- script to generate graphs, tables over experiment seeds
- A3C
- implement so scobi returns a correct supported environments list
- correct env.observation_space after env init (probably by constructing the f-vector with dummy objects)
### Minor
- split up focus.py
- rework terminal output with python logging

## How to simple call
```bash
# for training
python train.py --config path/to/config/file.yaml

# for showing the liveplot while using a trained model
python eval.py --config configs/gen-boxing.yaml liveplot True
```


## A little deeper explanation about the framework

The main file to call in this framework is `src/train.py` for training, and `src/eval.py` for evaluation. It is called together with a path to a config file. The structure of the config files are defined in `src/xrl/xrl_config.py`. A config file configures the Atari-Game, the selected algorithms and methods for the modular blocks and other training parameters.

plus focus files

Parameters set in the config file can be overwritten by calling the parameter and its new value in the command. For example:

```bash
python src/train.py --config path/to/config/file.yaml device "cpu"
```
