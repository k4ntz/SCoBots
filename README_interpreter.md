# Interpreter Decision Tree Extraction and Code Generation Tools

This repository contains tools for extracting interpretable decision trees from neural network policies using the Interpreter algorithm, and converting those trees into executable Python code. Unlike VIPER, the Interpreter approach focuses on using only original features from the environment and automatically generating feature combinations through pairwise differences.

## Key Features

- **Denormalized Environment Support (default)**: Works best with denormalized environments, allowing for more accurate feature combination calculations.
- **Automatic Feature Filtering**: Filters out combined features from the environment and focus files, using only original features.
- **Pairwise Difference Combinations**: Automatically generates feature combinations through pairwise differences as described in the Interpreter paper.
- **Decision Tree Extraction**: Trains decision trees that can approximate the behavior of neural network policies.
- **Code Generation**: Converts trained decision trees into readable Python code.

## Tools Overview

### 1. `interpreter_extract.py`

This tool implements the Interpreter algorithm to extract decision trees from neural network policies. It focuses on:

- Filtering environment and focus file features to obtain only raw/original features 
- Using denormalized observations for better accuracy
- Training decision trees on the filtered features with pairwise differences

#### Usage

```bash
python interpreter_extract.py -i <checkpoint_folder> -r interpreter [options]
```

#### Required Arguments

- `-i, --input`: Checkpoint folder name containing 'best_model.zip' and 'best_vecnormalize.pkl'
- `-r, --rule_extraction`: Rule extraction method to use (currently only supports "interpreter")

#### Optional Arguments

- `-e, --episodes`: Number of episodes to evaluate agent samples on (default: 5)
- `-n, --name`: Experiment name (default: "extraction")

#### Example

```bash
python interpreter_extract.py -i resources/checkpoints/Pong_seed0_reward-human_oc_pruned -r interpreter -e 10
```

### 2. `tree2code.py`

This tool converts trained decision trees from the Interpreter algorithm into executable Python code. Unlike VIPER's code generation, it:

- Creates code that directly uses the raw features from the environment
- Automatically adds code for the pairwise differences needed by the decision tree
- Generates human-readable variable mappings

#### Usage

```bash
python tree2code.py -i <input_folder> -g <game_name> -ff <focus_file> [options]
```

#### Required Arguments

- `-i, --input`: Folder name containing the trained decision tree (.pkl file)
- `-g, --game`: Game name (e.g., "Pong", "Breakout")
- `-ff, --focus_file`: Focus file path to match feature filtering

#### Optional Arguments

- `-n, --name`: Input filename (if not specified, will use the file with the most leaves)
- `-o, --output`: Output filename (if not specified, will be generated based on leaf count)

#### Example

```bash
python tree2code.py -i resources/interpreter_extract/extract_output/Pong_seed0_reward-human_oc_pruned-extraction -g Pong -ff resources/focusfiles/pruned_pong.yaml
```

## How It Works

1. **Feature Filtering**: The Interpreter approach first filters out combined features from the environment, keeping only raw features like positions and dimensions.

2. **Automatic Feature Combination**: Instead of using pre-defined combined features from the environment, it automatically generates pairwise differences between features (e.g., `ball.x - player1.x`).

3. **Denormalized Data**: For best results, the algorithm uses denormalized observations from the environment. This allows the decision tree to work with the actual values instead of normalized ones, which can distort the feature combination calculations.

4. **Decision Tree Training**: The algorithm trains a decision tree to approximate the neural network policy, using a DAgger-like approach to iteratively collect data and refine the tree.

5. **Code Generation**: The `tree2code.py` tool converts the trained decision tree into Python code, including the necessary pairwise difference calculations.

## Differences from VIPER

- **Feature Source**: Interpreter uses only raw features and generates its own combinations, while VIPER can use pre-calculated combined features from the environment.
- **Combination Method**: Interpreter uses pairwise differences between features, while VIPER relies on predefined feature combinations.

## Output

The `tree2code.py` tool generates Python code files in the `resources/program_policies/<input_folder_name>/` directory. The generated code includes:

- Feature extraction and mapping with meaningful variable names
- Pairwise difference calculations
- Decision tree logic converted to if-else statements
- A `play()` function that can be used directly with the environment

## Notes

1. For better results, use **denormalized** observations from the environment.
2. The tool automatically filters out combined features from the environment using the auto_generate_mask function.
3. Make sure to use the same focus file for both extraction and code generation to maintain consistency in feature filtering. 