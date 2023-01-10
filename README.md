# scobi
Successive Concepet Bottleneck Interface


TODO: SeSz REWORK ME

## What is happening here?
you need ocatari

raw -> object bottleneck -> concept bottleneck -> env

silent mode:
```
from scobi.utils import logging
logging.SILENT = True
```

## How to use (example)

```
from scobi import Environment

# Minimal init, not interactive, no focus dir/file specified
env = Environment(env_name='PongDeterministic-v4')
env.reset()
obs, reward, truncated, terminated, info = env.step(1)
env.action_space                    # Discrete(6)
env.action_space_description        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
env.observation_space               # Box(-1000.0, 1000.0, (81,), float32)
env.observation_space_description   # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
env.close()

# Extensive init, interactive, custom fcous dir and focus file
env = Environment(env_name='PongDeterministic-v4', interactive=True, focus_dir="experiments/my_focusfiles", focus_file="pruned_pong.yaml")
env.reset()
obs, reward, truncated, terminated, info = env.step(1)
env.action_space                    # Discrete(4)
env.action_space_description        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
env.observation_space               # Box(-1000.0, 1000.0, (12,), float32)
env.observation_space_description   # [['POSITION', 'ball'], ['POSITION', 'enemy'], ['POSITION', 'player'], ...
env.close()
```
atm observation_space innacurate, but no implications for learning now
## How to simple call



```
# for training
python train.py --config path/to/config/file.yaml

# for showing the liveplot while using a trained model
python eval.py --config configs/gen-boxing.yaml liveplot True
```


## A little deeper explanation about the framework

The main file to call in this framework is `src/train.py` for training, and `src/eval.py` for evaluation. It is called together with a path to a config file. The structure of the config files are defined in `src/xrl/xrl_config.py`. A config file configures the Atari-Game, the selected algorithms and methods for the modular blocks and other training parameters.

plus focus files

Parameters set in the config file can be overwritten by calling the parameter and its new value in the command. For example:

```
python src/train.py --config path/to/config/file.yaml device "cpu"
```