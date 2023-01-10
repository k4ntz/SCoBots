# scobi
Successive Concepet Bottleneck Interface


TODO: SeSz REWORK ME

## What is happening here?
you need ocatari


raw -> object bottleneck -> concept bottleneck -> env

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