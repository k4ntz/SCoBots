# scobi
Successive Concepet Bottleneck Interface

you need ocatari

## How to use (example)

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


## PPO SCoBots

PPO Training
```bash
python -m baselines.train -g Pong -s 42 -c 8 --num_layers 2 --input_data OCAtari --adam_step_size 0.001 --prune no_prune
```
Extract Model
```bash
python -m baselines.eval -g Pong -s 42 -t 1 --num_layers 2 --input_data OCAtari --prune no_prune --save_model
```


Collect Training Samples for ECLAIRE
- Navigate to experiments folder: ```/experiments```
```bash
python -m eval --config-file configs/re-pong.yaml --eclaire-cfg-file ../eclaire_configs/config_eclaire_Pong_s42_re_pr-nop_OCAtariinput_1l-v3.yaml rl_algo 3
```

Rule Extraction via ECLAIRE
- Navigate back into to the root folder of repo: ```/```
- Activate the environment for ECLAIRE
```bash
python -m baselines.rules --eclaire-cfg-file eclaire_configs/config_eclaire_Pong_s42_re_pr-nop_OCAtariinput_12-v3.yaml
```

Evaluate Rule Based Policy
- Navigate to experiments folder: ```/experiments```
```bash
python evaluate.py -g Pong -s 42 -c 8 -r env
```

Alternatively, you can use the provided scripts in the ```/scripts_for_experiment_exec``` folder to execute the above commands for multiple configurations. Note that you be in the root folder of the repo (SCoBots) to execute the scripts.