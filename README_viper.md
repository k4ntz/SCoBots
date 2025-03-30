# VIPER Decision Tree to Python Code Tool

This tool converts VIPER decision tree models into executable Python code. It supports three different output formats, each optimized for different use cases.

## Features

`tree2code_viper.py` is an integrated tool that provides three code generation modes:

1. **Standard Mode**: Includes complete feature extraction and combined feature calculations, producing code that most closely matches the original VIPER implementation (**only works with denormalized Tree!!**).
2. **Fast Mode**: Uses direct state vector indexing to generate minimized high-performance code, suitable for environments requiring efficient execution.
3. **Readable Mode**: Replaces state indices with meaningful variable names, generating code that is easy to read and understand, ideal for debugging and analysis.


## Usage

Basic command format:

```bash
python tree2code_viper.py -i <input_dir> -g <game_name> -m <mode> [options]
```

### Required Arguments

- `-i, --input`: Input directory containing VIPER model files (`.viper` or `.pkl`)
- `-g, --game`: Game name, such as "Pong" or "Breakout"
- `-m, --mode`: Code generation mode, possible values: `standard`, `fast`, `readable`

### Optional Arguments

- `-n, --name`: Specify model filename (if not specified, will automatically select the model with the most leaves or highest score)
- `-o, --output`: Output filename (if not specified, will be automatically generated based on leaf count)
- `-ff, --focus_file`: Focus file path (optional)
- `--debug`: Enable debug output

## Examples

### Generate Standard Code

```bash
python tree2code_viper.py -i resources/viper_extracts/extract_output/Pong_seed0_reward-human_oc_pruned-extraction/viper_trees -g Pong -m standard
```

### Generate Fast Code

```bash
python tree2code_viper.py -i resources/viper_extracts/extract_output/Pong_seed0_reward-human_oc_pruned-extraction/viper_trees -g Pong -m fast
```

### Generate Readable Code

```bash
python tree2code_viper.py -i resources/viper_extracts/extract_output/Pong_seed0_reward-human_oc_pruned-extraction/viper_trees -g Pong -m readable
```

## Output Specification

The tool generates Python code files in the `resources/program_policies_viper/<input_dir_name>/` directory, with filenames following this format:

- Standard version: `play_viper_standard_<num_leaves>_leaves.py`
- Fast version: `play_viper_fast_<num_leaves>_leaves.py`
- Readable version: `play_viper_readable_<num_leaves>_leaves.py`

where `<num_leaves>` is the number of leaf nodes in the decision tree.

## Notes

1. Standard code recalculates combined features, only works good with **denormalized** Tree version.
2. Fast code assumes all features (including combined features) are directly available in the state vector, without performing any feature calculations.
3. Readable code uses meaningful variable names but still directly uses values from the state vector without recalculating combined features.
4. If no focus file is specified, the tool will attempt to get feature descriptions directly from the environment.

## Error Handling

If model loading fails or feature names cannot be retrieved, the tool provides appropriate error messages and fallback options. For example, when feature names cannot be retrieved, default generic feature names (`feature_0`, `feature_1`, etc.) will be used. 