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
python -m baselines.rules --eclaire-cfg-file eclaire_configs/config_eclaire_Pong_s42_re_pr-nop_OCAtariinput_1l-v3.yaml
```

Evaluate Rule Based Policy
- Navigate to experiments folder: ```/experiments```
```bash
python -m explain --config-file configs/re-pong.yaml --eclaire-cfg-file ../eclaire_configs/config_eclaire_Pong_s42_re_pr-ext_OCAtariinput_1l-v3.yaml rl_algo 3
```

Alternatively, you can use the provided scripts in the folders ```/scripts_scobots_env```, ```/scripts_remix_env```, ```/scripts_for_analysis```
to execute the above commands for multiple configurations. Note that you be in the root folder of the repo (SCoBots) to execute the scripts.

## Visualizing the Rules

The code for visualizing is in the folder ```/remix_visualization```. It is copied from the Remix repository (https://github.com/mateoespinosa/remix). The style.css file is just a dummy file. The visualization can be started by running the following command:
```bash
python -m remix_visualization.visualize_ruleset file_path_to_ruleset
```

## In Progress: Calculating Divergence between PPO policy and Rule Based Policy
This code is still experimental and in progress. It can be done using the scripts in the folder ```/scripts_kl_divergence```. The order should be as follows:
1. Collect samples for the PPO policy
2. Get rule set policy output for the samples
3. Calculate the disagreement between the two policies


## Data
The data for this repo can be downloaded (https://hessenbox.tu-darmstadt.de/getlink/fi3BrmgYx9JyN54FokGkXwEQ/), but you need to get permission to access the data first. The data includes the following:
- baselines_checkpoints_s42_final.tar.gz:
    - Contains the final checkpoints for the PPO models trained on Pong and Boxing
    - The data should be placed in a directory called ```baselines_checkpoints``` in the root directory of the repo.
- eclaire_results.tar.gz:
    - Contains the results of the ECLAIRE rule extraction for Pong and Boxing
    - The data should be placed in the root directory of the repo.
- scobots_spaceandmoc_detectors.tar.gz:
    - Contains the SPACE+MOC object detectors for Pong and Boxing
    - This data must be used in conjunction with the dev branch of the repo https://github.com/nlsgrndn/SCoBots:
        - That repo should be cloned into the same parent directory as this repo and renamed to "space_and_moc". Note that you might have temporary rename the SCoBots repo to something else while cloning the space_and_moc repo because the folder names are the same initially.
        The final folder structure should look like this:
        ```
        parent_directory
        |_ SCoBots (space_detectors branch of this repo)
        |_ space_and_moc (dev branch of https://github.com/nlsgrndn/SCoBots)
        ```
        - Follow the setup instructions in the space_and_moc repo.

## Demonstration of SCoBot with SPACE+MOC Object Detectors and ECLAIRE policy
- Navigate to experiments folder: ```/experiments```
```bash
python -m explain --config-file configs/re-pong.yaml --eclaire-cfg-file ../eclaire_configs/config_eclaire_Pong_s42_re_pr-ext_SPACEinput_2l-v3
.yaml rl_algo 3
```
