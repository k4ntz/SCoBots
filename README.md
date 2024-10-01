# Successive Concepet Bottleneck Interface SCoBot
## Requirements
Scobots requires OCAtari and the local var ```'SCOBI_OBJ_EXTRACTOR'``` set as either ```OC_Atari``` or ```Noisy_OC_Atari``` as well as every dependency listed in the ```requirements.txt```.

Without agents SCoBots are not usable, so you can either download some pretrained agents from huggingface via the ```download_agents.sh``` script, or train one yourself, as explained in the usage guide.

## Usage
There are three python files which can be run directly
1. train.py
2. eval.py
3. render_agent.py

### Training an agent
Execute the train.py file to train an agent for a specific game, with a specified amount of cores and a specified seed.
The file is executable like described in the following with these flags:
```bash
python train.py -g -s -env -r -x --rgbv4 --rgbv5
```
The first three flags are required as input.

The content of the flags can be found in the following code block:
```bash
-g game (e.g. -g Pong, -g Skiing, ...)
-s seed (e.g. -s 0, -s 42, ...)
-env vect. environments (number of env used e.g. -c 1, -c 5, ...) the standard value is 8
-r reward (what reward model with options env, human, mixed)
-x exclude-properties (properties to be excluded from the vector)
--rgbv4 set the observation space as rgbv4
--rgbv5 set the observation space as rgbv5
```
### Evaluating an agent
The evaluate.py file evaluates an already trained agent, displaying the results afterwards and saving it in a dedicated file.

The file is executable like described in the following with these flags:
```bash
python eval.py -g -s -t -r -p -x --rgb
```
The first three flags are required as input.

The content of the flags can be found in the following code block:
```bash
-g game (e.g. -g Pong, -g Skiing, ...)
-s seed (e.g. -s 0, -s 42, ...)
-t times (how many steps the eval uses e.g. -t 1, -t 100, ...)
-r reward (what reward model with options -r env, -r human, -r mixed)
-p prune (used if desire to use a pruned focusfile with options -p internal, -p external)
-x exclude-properties (properties to be excluded from the vector)
--rgb set the observation space as rgb instead of obj centric
```

### Watching a trained agent
To visualize a trained agent playing a specified game the render_agent.py file can be executed.
Running the file will open and display the game played as a gif.

The file is executable like described in the following with these flags:
```bash
python render_agent.py -g -s -t -r -p -x --rgb
```
The first three flags are required as input.

The content of the flags can be found in the following code block:
```bash
-g game (e.g. -g Pong, -g Skiing, ...)
-s seed (e.g. -s 0, -s 42, ...)
-t times (how many steps the eval uses e.g. -t 1, -t 100, ...)
-r reward (what reward model with options -r env, -r human, -r mixed)
-p prune (used if desire to use a pruned focusfile with options -p internal, -p external)
-x exclude-properties (properties to be excluded from the vector)
--rgb set the observation space as rgb instead of obj centric
```

## Usage of Checkpoints
Checkpoints are saved under ```checkpoints```.
Each folder states in its name explicitly the specifications given for the training.
So e.g. the folder ```Pong_seed0_reward-human-v2``` denotes that the trained agent was trained with a ```seed``` of 0, its reward model is the ```human``` option.
So a usage of ```eval.py``` with this agent would look like ```python eval.py -g Pong -s 0 -r human``` or ```python render_agent.py -g Pong -s 0 -r human```

With the checkpoint being stored accordingly named in the checkpoints folder, it will automaticlly be loaded and there is no need to provide an explicit storage path.

Furthermore during the training process every 10th epoch will be saved via pickle to ensure less frustrating training process in regards of potential crashes. These are saved separately in a sub-folder named ```training_checkpoints``` next to the ```best_model.zip``` and ```best_vecnormalize.pkl``` which are saved after a complete successful training process in. 
