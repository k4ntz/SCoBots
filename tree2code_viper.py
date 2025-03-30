#!/usr/bin/env python
"""
Tree to Code Tool - Combined Version
Convert VIPER decision trees to Python code in three different formats:
1. Standard - Full implementation with feature combination calculations
2. Fast - High-performance implementation with direct state indexing
3. Readable - Easy-to-read implementation with meaningful variable names

Usage:
    python tree2code_combined.py -i <input_dir> -g <game_name> -m <mode> [options]
    
Mode selection:
    -m standard: Standard version, supports feature combination calculations
    -m fast: Fast version, uses direct state indexing
    -m readable: Readable version, uses meaningful variable names
"""

import argparse
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from joblib import load
from sklearn.tree import _tree

# Define a small constant to avoid division by zero
EPS = np.finfo(np.float64).eps.item()

def load_model(input_dir: str, model_name: Optional[str] = None) -> Tuple[object, int]:
    """
    Load VIPER model from input directory
    
    Args:
        input_dir: Directory containing VIPER model files
        model_name: Optional model filename
        
    Returns:
        tuple: (model object, number of leaf nodes)
    """
    try:
        best_model_path = None
        
        if model_name is None:
            # First try to find viper files with "best" in their name
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".viper") and "_best" in file_name:
                    best_model_path = os.path.join(input_dir, file_name)
                    break
            
            # If no best model found, try using Tree-* pattern files, sorted by score
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
            
            # If still not found, find the model with the most leaves
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
                raise FileNotFoundError(f"No usable model found in {input_dir}")
            
            model_path = best_model_path
        else:
            model_path = os.path.join(input_dir, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_path} not found")
        
        model = load(model_path)
        
        num_leaves = count_leaves(model.tree_)
        print(f"Successfully loaded model from {model_path} with {num_leaves} leaf nodes")
        return model, num_leaves
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

def count_leaves(tree) -> int:
    """
    Count the number of leaf nodes in a decision tree
    
    Args:
        tree: Decision tree object
        
    Returns:
        Number of leaf nodes
    """
    return sum(1 for _ in range(tree.node_count) if tree.children_left[_] == -1)

def sanitize_var_name(feature_name: str) -> str:
    """
    Convert feature names to valid Python variable names
    
    Args:
        feature_name: Feature name
        
    Returns:
        Valid Python variable name
    """
    # Handle feature names with parentheses and other structures, maintaining consistent variable name style
    # First check if it's a combined feature
    if '(' in feature_name and ')' in feature_name:
        # Extract feature type and objects
        match = re.match(r'([A-Z]+)\(([^,]+),\s*([^)]+)\)(\.([^.]+))?', feature_name)
        if match:
            feature_type, obj1, obj2, _, axis = match.groups()
            if axis:
                return f"{feature_type}_{obj1.strip()}_{obj2.strip()}_{axis}"
            else:
                return f"{feature_type}_{obj1.strip()}_{obj2.strip()}"
        
        # Handle single object features
        match = re.match(r'([A-Z]+)\(([^)]+)\)(\.([^.]+))?', feature_name)
        if match:
            feature_type, obj, _, axis = match.groups()
            if axis:
                return f"{feature_type}_{obj.strip()}_{axis}"
            else:
                return f"{feature_type}_{obj.strip()}"
    
    # Handle time features
    name = re.sub(r'\[t-(\d+)\]', r'_prev', feature_name)
    
    # Handle dots (replace with underscores)
    name = name.replace('.', '_')
    
    # Replace spaces and other illegal characters
    name = name.replace(' ', '_')
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Ensure variable name doesn't start with a digit
    if name and name[0].isdigit():
        name = 'f_' + name
    
    # If variable name is empty, use default
    if not name:
        name = "unknown_feature"
        
    return name

def process_combined_features(feature_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Process feature list, distinguish between original features and combined features,
    inline calculation logic directly into the code
    
    Args:
        feature_names: List of feature descriptions
        
    Returns:
        Tuple[List[str], Dict[str, str]]: (original feature list, combined features dict{name: calculation code})
    """
    # Distinguish between original and combined features
    original_features = []
    combined_features = {}
    
    for feature in feature_names:
        if "(" in feature:  # Combined feature
            # Generate combined feature code based on feature type
            if feature.startswith("D("):
                # Distance feature: D(obj1, obj2).axis
                match = re.match(r"D\(([^,]+),\s*([^\)]+)\)\.([xy])", feature)
                if match:
                    obj1, obj2, axis = match.groups()
                    obj1, obj2 = obj1.strip(), obj2.strip()
                    
                    # Inline distance calculation logic
                    if axis == 'x':
                        combined_features[feature] = f"{obj2}_x - {obj1}_x"
                    else:  # axis == 'y'
                        combined_features[feature] = f"{obj2}_y - {obj1}_y"
            
            elif feature.startswith("ED("):
                # Euclidean distance: ED(obj1, obj2)
                match = re.match(r"ED\(([^,]+),\s*([^\)]+)\)", feature)
                if match:
                    obj1, obj2 = match.groups()
                    obj1, obj2 = obj1.strip(), obj2.strip()
                    
                    # Inline Euclidean distance calculation logic
                    combined_features[feature] = f"math.sqrt(({obj2}_y - {obj1}_y)**2 + ({obj2}_x - {obj1}_x)**2)"
            
            elif feature.startswith("C("):
                # Center point: C(obj1, obj2).axis
                match = re.match(r"C\(([^,]+),\s*([^\)]+)\)\.([xy])", feature)
                if match:
                    obj1, obj2, axis = match.groups()
                    obj1, obj2 = obj1.strip(), obj2.strip()
                    
                    # Inline center point calculation logic
                    if axis == 'x':
                        combined_features[feature] = f"({obj1}_x + {obj2}_x) / 2"
                    else:  # axis == 'y'
                        combined_features[feature] = f"({obj1}_y + {obj2}_y) / 2"
            
            elif feature.startswith("V("):
                # Velocity: V(obj).axis
                match = re.match(r"V\(([^\)]+)\)\.([xy])", feature)
                if match:
                    obj, axis = match.groups()
                    obj = obj.strip()
                    
                    # Inline velocity calculation logic
                    combined_features[feature] = f"math.sqrt(({obj}_x_prev - {obj}_x)**2 + ({obj}_y_prev - {obj}_y)**2)"
            
            elif feature.startswith("DV("):
                # Directional velocity: DV(obj).axis
                match = re.match(r"DV\(([^\)]+)\)\.([xy])", feature)
                if match:
                    obj, axis = match.groups()
                    obj = obj.strip()
                    
                    # Inline directional velocity calculation logic
                    if axis == 'x':
                        combined_features[feature] = f"{obj}_x_prev - {obj}_x"
                    else:  # axis == 'y'
                        combined_features[feature] = f"{obj}_y_prev - {obj}_y"
            
            elif feature.startswith("LT("):
                # Linear trajectory: LT(obj1, obj2).axis
                match = re.match(r"LT\(([^,]+),\s*([^\)]+)\)\.([xy])", feature)
                if match:
                    obj1, obj2, axis = match.groups()
                    obj1, obj2 = obj1.strip(), obj2.strip()
                    
                    # Inline linear trajectory calculation logic
                    # First calculate slope and intercept
                    slope = f"({obj2}_y_prev - {obj2}_y) / ({obj2}_x_prev - {obj2}_x + 0.1)"
                    intercept = f"{obj2}_y - {slope} * {obj2}_x"
                    
                    if axis == 'x':
                        # distx = ((a_position[1] - b) / (m + EPS)) - a_position[0]
                        combined_features[feature] = f"(({obj1}_y - {intercept}) / ({slope} + {EPS})) - {obj1}_x"
                    else:  # axis == 'y'
                        # disty = (m * a_position[0] + b) - a_position[1]
                        combined_features[feature] = f"({slope} * {obj1}_x + {intercept}) - {obj1}_y"
        else:
            original_features.append(feature)
    
    return original_features, combined_features

def generate_feature_extraction_code(feature_names: List[str]) -> Tuple[str, Dict[int, str]]:
    """
    Generate code to extract features from the state vector
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Tuple[str, Dict[int, str]]: Feature extraction code and mapping from feature indices to variable names
    """
    code = ["    # Extract features from state vector (using original values, not recalculating)"]
    
    # Mapping from feature indices to variable names
    var_name_map = {}
    
    # Track duplicate variables to avoid name conflicts
    used_var_names = {}
    
    # Generate variable definitions for each feature
    for i, feature in enumerate(feature_names):
        var_name = sanitize_var_name(feature)
        
        # Handle duplicate variable names
        if var_name in used_var_names:
            var_name_map[i] = used_var_names[var_name]
            continue
        
        # Record that this variable name has been used and save the corresponding index
        used_var_names[var_name] = var_name
        var_name_map[i] = var_name
        
        # Generate code
        code.append(f"    {var_name} = state[{i}]")
    
    return "\n".join(code), var_name_map

def load_feature_names_from_env(game: str, focus_file: Optional[str] = None) -> List[str]:
    """
    Load feature names from the environment
    
    Args:
        game: Game name
        focus_file: Focus file path (optional)
        
    Returns:
        List of feature names
    """
    try:
        from scobi import Environment
        
        # Create environment instance
        env_str = "ALE/" + game + "-v5"
        
        # If a focus file is provided, use it
        if focus_file:
            focus_dir = focus_file.rsplit("/", 1)[0]
            focus_file_name = focus_file.split("/")[-1]
            
            env = Environment(env_str,
                            focus_dir=focus_dir,
                            focus_file=focus_file_name,
                            hide_properties=False,
                            draw_features=True,
                            reward=0)
        else:
            # Otherwise, don't use a focus file
            env = Environment(env_str,
                            hide_properties=False,
                            draw_features=True,
                            reward=0)
        
        # Get feature descriptions from the environment
        feature_descriptions = env.get_vector_entry_descriptions()
        print(f"Loaded {len(feature_descriptions)} features from environment")
        print("Environment feature list:")
        for i, desc in enumerate(feature_descriptions):
            print(f"  {i}: {desc}")
        
        return feature_descriptions
        
    except Exception as e:
        print(f"Failed to load feature names from environment: {e}")
        traceback.print_exc()
        return []

def find_focus_file(input_dir: str, game: str) -> Optional[str]:
    """
    Find focus file based on the following priority:
    1. Focus files in the model directory
    2. Game-specific focus files in resources/focusfiles
    3. Default focus file
    
    Args:
        input_dir: Model directory
        game: Game name
        
    Returns:
        Path to the focus file, or None if not found
    """
    # Try to find focus files in the model directory
    focus_files = []
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            if file.lower().endswith(".yaml"):
                focus_files.append(os.path.join(input_dir, file))
    
    if focus_files:
        print(f"Found focus file in model directory: {focus_files[0]}")
        return focus_files[0]
    
    # Look for game-specific focus files in resources/focusfiles
    game_name_lower = game.lower()
    focus_dir = "resources/focusfiles"
    if os.path.exists(focus_dir):
        for file in os.listdir(focus_dir):
            if file.lower().endswith(".yaml") and game_name_lower in file.lower():
                focus_file = os.path.join(focus_dir, file)
                print(f"Found game-specific focus file in resources/focusfiles: {focus_file}")
                return focus_file
    
    # Try to find pruned_ version focus files in resources/focusfiles
    if os.path.exists(focus_dir):
        for file in os.listdir(focus_dir):
            if file.lower().startswith("pruned_") and file.lower().endswith(".yaml"):
                focus_file = os.path.join(focus_dir, file)
                print(f"Found pruned version focus file in resources/focusfiles: {focus_file}")
                return focus_file
    
    # Look for default focus files
    default_focus_paths = [
        "resources/focusfiles/default.yaml",
        "paper_experiments/focusfiles/default.yaml",
        "paper_experiments/norel_focusfiles/default.yaml"
    ]
    
    for path in default_focus_paths:
        if os.path.exists(path):
            print(f"Using default focus file: {path}")
            return path
    
    print("No focus file found")
    return None

# =========================== Standard Version Implementation ===========================

def tree_to_code_standard(tree, feature_names: List[str], combined_features: Optional[Dict[str, str]] = None) -> str:
    """
    Convert decision tree to Python if-else logic code
    
    Args:
        tree: Decision tree object
        feature_names: List of feature names
        combined_features: Combined features dictionary {feature name: calculation code}
        
    Returns:
        Python logic code string
    """
    combined_features = combined_features or {}
    tree_ = tree.tree_
    output = []
    
    # Check if feature count matches
    n_features = tree_.n_features
    if len(feature_names) != n_features:
        print(f"Warning: Number of feature names ({len(feature_names)}) does not match number of features in the tree ({n_features})")
        if len(feature_names) < n_features:
            # Fill in missing parts with default feature names
            feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), n_features)]
    
    def recurse(node, depth):
        """
        Recursively generate Python code for decision tree nodes
        """
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_index = tree_.feature[node]
            threshold = tree_.threshold[node]
            
            if feature_index >= len(feature_names):
                print(f"Warning: Feature index {feature_index} is out of range for feature names list {len(feature_names)}")
                feature_code = f"state[{feature_index}]"
                output.append(f"{indent}if {feature_code} <= {threshold:.6f}:")
            else:
                # Get the feature name
                feature_name = feature_names[feature_index]
                
                # If it's a combined feature, use its calculation code
                if feature_name in combined_features:
                    feature_code = combined_features[feature_name]
                    output.append(f"{indent}if {feature_code} <= {threshold:.6f}:")
                else:
                    # Original feature, use variable name directly
                    variable_name = sanitize_var_name(feature_name)
                    output.append(f"{indent}if {variable_name} <= {threshold:.6f}:")
                
            recurse(tree_.children_left[node], depth + 1)
            output.append(f"{indent}else:")
            recurse(tree_.children_right[node], depth + 1)
        else:
            action = np.argmax(tree_.value[node])
            output.append(f"{indent}return {action}")

    recurse(0, 1)
    return "\n".join(output)

# =========================== Fast Version Implementation ===========================

def tree_to_code_fast(tree) -> str:
    """
    Convert decision tree to minimal Python code using direct state indexing
    
    Args:
        tree: Decision tree object
        
    Returns:
        String containing minimal Python code implementing the decision tree
    """
    tree_ = tree.tree_
    
    # Define function header without any feature definitions
    code = ["def play(state):"]
    
    # Recursive function to build code for a single node
    def recurse(node, depth):
        indent = "    " * (depth + 1)
        
        if tree_.feature[node] != -2:  # Non-leaf node
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            
            # Use direct state indexing for all features
            condition = f"state[{feature_idx}] <= {threshold}"
            
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            
            code.append(f"{indent}if {condition}:")
            recurse(left_child, depth + 1)
            
            code.append(f"{indent}else:")
            recurse(right_child, depth + 1)
        else:  # Leaf node
            class_values = tree_.value[node][0]
            class_idx = np.argmax(class_values)
            code.append(f"{indent}return {class_idx}")
    
    # Start recursion from the root node
    recurse(0, 0)
    
    return "\n".join(code)

# =========================== Readable Version Implementation ===========================

def tree_to_code_readable(tree, feature_names: List[str] = None) -> str:
    """
    Convert decision tree to Python code with meaningful variable names
    
    Args:
        tree: Decision tree object
        feature_names: List of feature names
        
    Returns:
        Python code string
    """
    tree_ = tree.tree_
    
    # Check if feature names are sufficient
    n_features = tree_.n_features
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    elif len(feature_names) < n_features:
        print(f"Warning: Number of feature names ({len(feature_names)}) is less than the number of features used by the tree ({n_features}), filling with generic names")
        feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), n_features)]
    
    # Generate feature extraction code
    feature_extraction_code, var_name_map = generate_feature_extraction_code(feature_names)
    
    # Define function header
    code = [
        "import numpy as np",
        "",
        "def play(state):",
        feature_extraction_code,
        ""
    ]
    
    # Recursively generate code for decision tree nodes
    def recurse(node, depth):
        indent = "    " * (depth + 1)
        
        if tree_.feature[node] != -2:  # Non-leaf node
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            
            # Replace feature indices with meaningful variable names
            var_name = var_name_map.get(feature_idx)
            if var_name is None:
                # If no mapping is found, use feature index
                var_name = f"state[{feature_idx}]"
                print(f"Warning: Feature index {feature_idx} has no corresponding variable name, using state[{feature_idx}] directly")
            
            condition = f"{var_name} <= {threshold}"
            
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            
            code.append(f"{indent}if {condition}:")
            recurse(left_child, depth + 1)
            
            code.append(f"{indent}else:")
            recurse(right_child, depth + 1)
        else:  # Leaf node
            class_values = tree_.value[node][0]
            class_idx = np.argmax(class_values)
            code.append(f"{indent}return {class_idx}")
    
    # Start recursion from the root node
    recurse(0, 0)
    
    # Add default return
    code.append("    return -1  # Default return -1")
    
    return "\n".join(code)

def main():
    parser = argparse.ArgumentParser(description="Convert VIPER decision tree to Python code (Combined Version)")
    parser.add_argument("-i", "--input", help="Input directory containing VIPER model files", required=True)
    parser.add_argument("-n", "--name", help="Input filename (optional, if not specified will use model with most leaves)")
    parser.add_argument("-o", "--output", help="Output filename (optional, will be generated based on leaf count)")
    parser.add_argument("-g", "--game", help="Game name", required=True)
    parser.add_argument("-ff", "--focus_file", help="Focus file path (optional, if not specified will search automatically)")
    parser.add_argument("-m", "--mode", help="Code generation mode: standard, fast, readable", default="standard", choices=["standard", "fast", "readable"])
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Load model
    model, num_leaves = load_model(args.input, args.name)
    
    # Determine focus file location
    focus_file = args.focus_file
    if not focus_file:
        focus_file = find_focus_file(args.input, args.game)
        if focus_file:
            print(f"Automatically found focus file: {focus_file}")
    
    # Try to load feature names
    feature_names = []
    if args.game:
        try:
            feature_names = load_feature_names_from_env(args.game, focus_file)
        except Exception as e:
            print(f"Error loading feature names from environment: {e}")
            if args.debug:
                traceback.print_exc()
    
    # If unable to load feature names from environment, use default feature names
    if not feature_names:
        feature_count = model.tree_.n_features
        print(f"Unable to load feature names from environment, using default feature names feature_0 to feature_{feature_count-1}")
        feature_names = [f"feature_{i}" for i in range(feature_count)]
    
    if args.debug:
        print(f"Number of features used by model: {model.tree_.n_features}")
        print(f"Number of available feature names: {len(feature_names)}")
        
        print("\nFeature names:")
        for i, name in enumerate(feature_names):
            print(f"  {i}: {name}")
    
    # Create output directory
    input_dir_name = os.path.basename(os.path.normpath(args.input))
    output_dir = os.path.join("resources", "program_policies_viper", input_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose code generation mode
    if args.mode == "standard":
        # Process combined features
        _, combined_features = process_combined_features(feature_names)
        
        if args.debug:
            print("\nCombined feature calculations:")
            for feat, code in combined_features.items():
                print(f"  {feat}: {code}")
        
        # Generate standard code
        feature_extraction_code = generate_feature_extraction_code(feature_names)[0]
        code_body = tree_to_code_standard(model, feature_names, combined_features)
        
        # Import necessary libraries
        imports = (
            "import numpy as np\n"
            "import math\n\n"
            "# Define a small constant to avoid division by zero\n"
            "EPS = np.finfo(np.float64).eps.item()\n\n"
        )
        
        # Build complete code
        code = imports + "def play(state):\n" + feature_extraction_code + "\n\n" + code_body
        
        # Generate output filename
        if args.output:
            output_file = os.path.join(output_dir, args.output)
        else:
            output_file = os.path.join(output_dir, f"play_viper_standard_{num_leaves}_leaves.py")
        
        mode_type = "Standard version (supports feature combinations)"
        
    elif args.mode == "fast":
        # Generate fast code
        code = tree_to_code_fast(model)
        
        # Generate output filename
        if args.output:
            output_file = os.path.join(output_dir, args.output)
        else:
            output_file = os.path.join(output_dir, f"play_viper_fast_{num_leaves}_leaves.py")
        
        mode_type = "High-performance version (direct indexing)"
        
    elif args.mode == "readable":
        # Generate readable code
        code = tree_to_code_readable(model, feature_names)
        
        # Generate output filename
        if args.output:
            output_file = os.path.join(output_dir, args.output)
        else:
            output_file = os.path.join(output_dir, f"play_viper_readable_{num_leaves}_leaves.py")
        
        mode_type = "Readable version (meaningful variable names)"
    
    # Write code to file
    with open(output_file, "w") as f:
        f.write(code)
    
    print(f"Successfully generated {mode_type} code and saved to {output_file}")
    print(f"This model uses {model.tree_.n_features} features")

if __name__ == "__main__":
    main() 