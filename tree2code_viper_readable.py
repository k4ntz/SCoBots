#!/usr/bin/env python
'''
Tree to Code Tool - Readable Version
For converting VIPER decision trees to Python code with meaningful variable names
This version directly uses state indices, but uses meaningful variable names instead of indices
'''

import argparse
import os
import re
import sys
import traceback
from typing import Dict, List, Tuple

import numpy as np
from joblib import load

def load_model(input_dir, model_name=None):
    """Load a VIPER model from the input directory."""
    if model_name is None:
        best_model_path = None
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".viper") and "_best" in file_name:
                best_model_path = os.path.join(input_dir, file_name)
                break
            
        if best_model_path is None:
            tree_files = []
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".viper") and file_name.startswith("Tree-"):
                    try:
                        score_match = re.search(r"Tree-\d+_(\d+\.?\d*)\.viper", file_name)
                        if score_match:
                            score = float(score_match.group(1))
                            tree_files.append((file_name, score))
                    except Exception:
                        continue
            
            if tree_files:
                tree_files.sort(key=lambda x: x[1], reverse=True)
                best_model_path = os.path.join(input_dir, tree_files[0][0])
        
        if best_model_path is None:
            max_leaves = 0
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".pkl") or file_name.endswith(".viper"):
                    match = re.search(r"(\d+)_leaves", file_name)
                    if match:
                        leaves = int(match.group(1))
                        if leaves > max_leaves:
                            max_leaves = leaves
                            best_model_path = os.path.join(input_dir, file_name)
        
        if best_model_path is None:
            raise FileNotFoundError(f"No available model files found in {input_dir}")
        
        model_path = best_model_path
    else:
        model_path = os.path.join(input_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found")
    
    model = load(model_path)
    
    num_leaves = count_leaves(model.tree_)
    print(f"Successfully loaded model from {model_path} with {num_leaves} leaves")
    return model, num_leaves

def count_leaves(tree):
    """Count the number of leaves in a decision tree."""
    return sum(1 for _ in range(tree.node_count) if tree.children_left[_] == -1)

def sanitize_var_name(feature_name: str) -> str:
    """
    Convert feature name to a valid Python variable name
    
    Args:
        feature_name: feature name
        
    Returns:
        valid Python variable name
    """
    # special processing of feature names with parentheses and other structures, keep consistent variable name style
    # first check if it is a combined feature
    if '(' in feature_name and ')' in feature_name:
        # extract feature type and objects
        match = re.match(r'([A-Z]+)\(([^,]+),\s*([^)]+)\)(\.([^.]+))?', feature_name)
        if match:
            feature_type, obj1, obj2, _, axis = match.groups()
            if axis:
                return f"{feature_type}_{obj1.strip()}_{obj2.strip()}_{axis}"
            else:
                return f"{feature_type}_{obj1.strip()}_{obj2.strip()}"
    
    # handle time features
    name = re.sub(r'\[t-(\d+)\]', r'_prev', feature_name)
    
    # handle dot (replace dot with underscore)
    name = name.replace('.', '_')
    
    # replace spaces and other illegal characters
    name = name.replace(' ', '_')
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # ensure the variable name does not start with a number
    if name[0].isdigit():
        name = 'f_' + name
        
    return name

def load_feature_names_from_env(game: str, focus_file: str) -> List[str]:
    """load the feature names from the environment"""
    try:
        import yaml
        from scobi import Environment
        
        # create environment instance
        env_str = "ALE/" + game + "-v5"
        focus_dir = focus_file.rsplit("/", 1)[0]
        focus_file_name = focus_file.split("/")[-1]

        env = Environment(env_str,
                        focus_dir=focus_dir,
                        focus_file=focus_file_name,
                        hide_properties=False,
                        draw_features=True,
                        reward=0)
        
        # get the feature descriptions from the environment
        feature_descriptions = env.get_vector_entry_descriptions()
        print(f"loaded {len(feature_descriptions)} features from the environment")
        print("environment feature list:")
        for i, desc in enumerate(feature_descriptions):
            print(f"  {i}: {desc}")
        
        return feature_descriptions
        
    except Exception as e:
        print(f"failed to load feature names from the environment: {e}")
        traceback.print_exc()
        return []


def generate_feature_extraction_code(feature_names: List[str]) -> Tuple[str, Dict[int, str]]:
    """
    generate the code to extract features from the state vector
    
    Args:
        feature_names: feature name list
        
    Returns:
        Tuple[str, Dict[int, str]]: feature extraction code and feature index to variable name mapping
    """
    code = ["    # extract features from the state vector (use the original values, do not recalculate)"]
    
    # feature index to variable name mapping
    var_name_map = {}
    
    # track duplicate variables, avoid variable name conflicts
    used_var_names = {}
    
    # generate variable definitions for each feature
    for i, feature in enumerate(feature_names):
        var_name = sanitize_var_name(feature)
        
        # handle duplicate variable names
        if var_name in used_var_names:
            # check the index pattern, for repeated features in the environment observation vector, they actually point to the same feature
            # here, reuse the previous variable name by recording the used variable names and indices when there are duplicates
            var_name_map[i] = used_var_names[var_name]
            continue
        
        # record that this variable name has been used, save the corresponding index
        used_var_names[var_name] = var_name
        var_name_map[i] = var_name
        
        # generate code
        code.append(f"    {var_name} = state[{i}]")
    
    return "\n".join(code), var_name_map

def tree_to_code_readable(tree, feature_names: List[str] = None) -> str:
    """
    convert the decision tree to Python code with meaningful variable names
    
    Args:
        tree: decision tree object
        feature_names: feature name list
        
    Returns:
        Python code string
    """
    tree_ = tree.tree_
    
    # check if the feature names are enough
    n_features = tree_.n_features
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    elif len(feature_names) < n_features:
        raise ValueError(f"error: the number of feature names ({len(feature_names)}) is less than the number of features used by the decision tree ({n_features}), cannot continue")
    
    # generate the feature extraction code
    feature_extraction_code, var_name_map = generate_feature_extraction_code(feature_names)
    
    # define the function header
    code = [
        "import numpy as np",
        "",
        "def play(state):",
        feature_extraction_code,
        ""
    ]
    
    # recursively generate the code for the decision tree nodes
    def recurse(node, depth):
        indent = "    " * (depth + 1)
        
        if tree_.feature[node] != -2:  # non-leaf node
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            
            # replace the feature index with a meaningful variable name
            var_name = var_name_map.get(feature_idx)
            if var_name is None:
                # if the mapping is not found, use the feature index
                var_name = f"state[{feature_idx}]"
                print(f"warning: the feature index {feature_idx} does not have a corresponding variable name, directly use state[{feature_idx}]")
            
            condition = f"{var_name} <= {threshold}"
            
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            
            code.append(f"{indent}if {condition}:")
            recurse(left_child, depth + 1)
            
            code.append(f"{indent}else:")
            recurse(right_child, depth + 1)
        else:  # leaf node
            class_values = tree_.value[node][0]
            class_idx = np.argmax(class_values)
            code.append(f"{indent}return {class_idx}")
    
    # recursively start from the root node
    recurse(0, 0)
    
    # add the default return
    code.append("    return -1  # default return -1")
    
    return "\n".join(code)

def main():
    parser = argparse.ArgumentParser(description='convert VIPER decision tree to readable Python code')
    parser.add_argument('-i', '--input', help='directory containing VIPER model files', required=True)
    parser.add_argument('-n', '--name', help='input file name (optional, will use the model with the most leaves if not specified)')
    parser.add_argument('-o', '--output', help='output file name (optional, will be generated based on the number of leaves)')
    parser.add_argument('-g', '--game', help='game name', required=True)
    parser.add_argument('-ff', '--focus_file', help='focus file path (optional)', required=False)
    parser.add_argument('--debug', action='store_true', help='enable debug output')
    
    args = parser.parse_args()
    
    # load the model
    model, num_leaves = load_model(args.input, args.name)
    
    # try to load the feature names
    feature_names = []
    if args.focus_file:
        # try to load the feature names from the environment
        feature_names = load_feature_names_from_env(args.game, args.focus_file)
    
    # if cannot load the feature names from the environment, use the default feature names
    if not feature_names:
        feature_count = model.tree_.n_features
        print(f"cannot load the feature names from the environment, use the default feature names feature_0 to feature_{feature_count-1}")
        feature_names = [f"feature_{i}" for i in range(feature_count)]
    
    if args.debug:
        print("feature names:")
        for i, name in enumerate(feature_names):
            print(f"  {i}: {name}")
            
        # check if the variable names are valid
        print("\ngenerated variable names:")
        for i, name in enumerate(feature_names):
            var_name = sanitize_var_name(name)
            print(f"  {i}: {name} -> {var_name}")
            
        # print the number of features used by the decision tree
        print(f"decision tree feature count: {model.tree_.n_features}")
    
    # convert the tree to code
    code = tree_to_code_readable(model, feature_names)
    
    # create the output directory
    input_dir_name = os.path.basename(os.path.normpath(args.input))
    output_dir = os.path.join("resources", "program_policies_viper", input_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # generate the output file name
    if args.output:
        output_file = os.path.join(output_dir, args.output)
    else:
        output_file = os.path.join(output_dir, f"play_viper_readable_{num_leaves}_leaves.py")
    
    # write the code to the file
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"successfully generated the readable code and saved to {output_file}")

if __name__ == "__main__":
    main()