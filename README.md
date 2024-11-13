# Successive Concept Bottleneck Agent SCoBot
## Installation And Requirements
Scobots needs OCAtari and the local var ```'SCOBI_OBJ_EXTRACTOR'``` set as either ```OC_Atari``` or ```Noisy_OC_Atari```. If not set it will automatically resort to ```OC_Atari```. Python version ```3.8.x``` is recommended if planning to use our RGB agents.

Without agents SCoBots are not usable, so you can either download some pre-trained agents from huggingface using the ```download_agents.sh``` script, or train one yourself, as explained in the usage-manual.

Due to issues with the ```autorom``` module versions, ```stable_baselines3[extras]``` has to be installed manually. The setup is completed with
```bash
pip install -r requirements.txt && pip install stable-baselines3[extras]==2.0.0
```

## How To Use
There are three Python files that can be run directly. Each of them has a ```-h``` help flag.

### Training An Agent
Execute the train.py file to train an agent for a given game, with a given number of cores and a specified seed.
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

### Monitoring A Trained Agent
To visualize a trained agent playing a specified game the render_agent.py file can be executed.
Running the file will open and display the game played as a gif.

The following example demonstrates the usage of the previously trained + evaluated agent:
```bash
python render_agent.py -g Pong -s 0 -r env
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
