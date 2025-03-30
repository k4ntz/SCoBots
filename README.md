# New Features

## Overview

This branch contains three main tool sets:

### 1. Decision Tree Extraction Tools

- **Interpreter**: Uses the Interpreter algorithm to automatically generate feature combinations through pairwise differences. [Details](README_interpreter.md)
- **VIPER**: Uses the VIPER algorithm to extract decision trees with denormalized enviroment.

### 2. Code Generation Tools

- **tree2code.py**: Converts Interpreter-extracted decision trees into Python code. [Details](README_interpreter.md)
- **tree2code_viper.py**: Converts VIPER-extracted decision trees into Python code in three different formats. [Details](README_viper.md)

### 3. Visualization Tools

- **render_agent.py**: Visualizes and interactively runs the generated Python policy code. [Details](README_render_pythonfile.md)

## Quick Start

### Interpreter Workflow
```bash
# 1. Extract decision tree
python interpreter_extract.py -i resources/checkpoints/Pong_seed0_reward-human_oc_pruned -r interpreter

# 2. Generate Python code
python tree2code.py -i resources/interpreter_extract/extract_output/Pong_seed0_reward-human_oc_pruned-extraction -g Pong -ff resources/focusfiles/pruned_pong.yaml

# 3. Visualize the generated Python code
python render_agent.py -g Pong -r human -s 0 -p default --python_file <path_to_generated_file>
```

### VIPER Workflow
```bash
# 1.(Optional) Extract decision tree using denormalized Env
python extract_viper_denormalized.py -g Pong -m resources/checkpoints/Pong_seed0_reward-human_oc_pruned -o resources/viper_denormalized -f pruned_pong.yaml

# 2. Generate Python code (from existing VIPER tree)
python tree2code_viper.py -i resources/viper_extracts/extract_output/Pong_seed0_reward-human_oc_pruned-extraction/viper_trees -g Pong -m readable

# 3. Visualize the generated Python code
python render_agent.py -g Pong -r human -s 0 -p default --python_file <path_to_generated_file>
```

## Key Features

- **Denormalized Environment Support**: Both methods perform best in denormalized environments
- **Execution Visualization**: Real-time view of code execution paths to understand decision processes
- **Multiple Generation Formats**: Includes standard, high-performance, and readable code generation formats
- **Automatic Feature Combination**: Interpreter automatically generates feature combinations through pairwise differences

## Detailed Documentation

- [Interpreter Extraction and Code Generation](README_interpreter.md)
- [VIPER Code Generation](README_viper.md)
- [Python Policy Visualization](README_render_pythonfile.md)

## Notes

1. Standard mode code works best in denormalized environments
2. To enable code execution visualization, modify the import statement in render_agent.py
3. Ensure extraction and code generation use the same focus file to maintain consistency
---
&nbsp;
# Original README
---
# Successive Concept Bottleneck Agent SCoBot
## Installation And Requirements
Scobots needs OCAtari and the local var ```'SCOBI_OBJ_EXTRACTOR'``` set as either ```OC_Atari``` or ```Noisy_OC_Atari```. If not set it will automatically resort to ```OC_Atari```. Python version ```3.8.x``` is recommended if planning to use our RGB agents.

Without agents SCoBots are not usable, so you can either download some pre-trained agents from huggingface using the ```download_agents.sh``` script, or train one yourself, as explained in the usage-manual.

Due to issues with the ```autorom``` module versions, ```stable_baselines3[extras]``` has to be installed manually. The setup is completed with
```bash
pip install -r requirements.txt && pip install "stable-baselines3[extras]==2.0.0"
```

Note that this version of SCoBots makes use of OC_Atari 2.0 and its neuro-symbolic state.

## How To Use
There are three Python files that can be run directly. Each of them has a ```-h``` help flag.

### Downloading Agents
The following commands will manually download and extract the agents to the ```resources``` folder.

For neural and tree-based agents:
```bash
# Download the agents (only seed0)
wget https://hessenbox.tu-darmstadt.de/dl/fi47F21YBzVZBRfGPKswumb7/resources_seed0.zip
unzip resources_seed0.zip
```
**or** 
```bash
# Download the agents (all seeds)
wget https://hessenbox.tu-darmstadt.de/dl/fiPLH36Zwi8EVv8JaLU4HpE2/resources_all.zip
unzip resources_all.zip
```

### Displaying A Trained Agent
To visualize a trained agent playing a specified game the render_agent.py file can be executed.
Running the file will open and display the game played as a gif.

The following example demonstrates the usage of the previously trained + evaluated agent:
```bash
python render_agent.py -g Pong -s 0 -r human -p default
```
Similar for decision-tree agents:
```bash
python render_agent.py -g Pong -s 0 -r human -p default --viper
```


### Training An Agent
Execute the ```train.py``` file to train an agent for a given game, with a given number of cores and a specified seed.
The following example demonstrates the usage:
```bash
python train.py -g Pong -s 0 -env 8 -r env --progress
```
The first three flags are required as input. With the help option the other flags can be displayed.
### Evaluating An Agent
The evaluate.py file evaluates an already trained agent, displaying the results afterwards and saving it in a dedicated file.

The following example demonstrates the usage of the previously trained agent:
```bash
python eval.py -g Pong -s 0 -t 10 -r env
```

## Usage Of Checkpoints And Example Workflow
Checkpoints are saved under ```resources/checkpoints```.
Each folder states in its name explicitly the training specifications.
So e.g. the folder ```Pong_seed0_reward-human_oc-n2``` denotes that the trained agent was trained with a ```seed``` of 0, its reward model is the ```human``` option, it is an ```object centered``` agent,  and that it is the second agent trained with these values.
So a usage with this agent would look like ```python eval.py -g Pong -s 0 -r human``` or ```python render_agent.py -g Pong -s 0 -r human```. This automatically picks the respectively latest trained agent named according to the values. For using a specific version the version flag has to be added.

With the checkpoint being stored accordingly named in the checkpoints folder, it will automaticlly be loaded and there is no need to provide an explicit storage path.

Unless explictily stated via ```--rgb```, it will always be automatically resorted to object centric checpoints.

Furthermore during the training process regularly checkpoints will be made and saved. These are saved separately in a sub-folder named ```training_checkpoints``` next to the ```best_model.zip``` and ```best_vecnormalize.pkl``` which are saved after a complete successful training process in. 

## Extracting Via Viper
If desired an extraction from a saved agent can be performed and saved under the folder ```viper_extracts```. An example usage would be:
```bash
python viper_extract.py -i Pong_seed0_reward-env_oc -r viper
```
Otherwise one can also hand a direct path after the ```-i``` flag. In this case though it is a MUST that the corresponding focusfile is correctly named inside of the given path next to the extracted tree.
The console prints what exactly the extractor is looking for.
