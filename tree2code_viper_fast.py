#!/usr/bin/env python
'''
Tree to Code Tool - High Performance Version
For converting VIPER decision trees to minimal Python code
This version directly uses state indices, without extracting features or recalculating combined features
'''

import argparse
import os
import re
import sys
import traceback
import numpy as np
from joblib import load

def load_model(input_dir, model_name=None):
    """Load a VIPER model from the input directory."""
    try:
        if model_name is None:
            # first try to find the viper file with best
            best_model_path = None
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".viper") and "_best" in file_name:
                    best_model_path = os.path.join(input_dir, file_name)
                    break
                
            # if no best model is found, try using the Tree-* mode files, sorted by score
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
                raise FileNotFoundError(f"no available model files found in {input_dir}")
            
            model_path = best_model_path
        else:
            model_path = os.path.join(input_dir, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"model {model_path} not found")
        
        model = load(model_path)
        
        num_leaves = count_leaves(model.tree_)
        print(f"successfully loaded model from {model_path} with {num_leaves} leaves")
        return model, num_leaves
    except Exception as e:
        print(f"error loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

def count_leaves(tree):
    """Count the number of leaves in a decision tree."""
    return sum(1 for _ in range(tree.node_count) if tree.children_left[_] == -1)

def tree_to_code_fast(tree) -> str:
    """Convert a decision tree to minimal Python code using direct state indexing.
    
    Args:
        tree: The decision tree object
        
    Returns:
        String containing the minimal Python code implementing the decision tree
    """
    tree_ = tree.tree_
    
    # Define function header without any feature definitions
    code = ["def play(state):"]
    
    # Recursive function to build the code for a single node
    def recurse(node, depth):
        indent = "    " * (depth + 1)
        
        if tree_.feature[node] != -2:  # Not a leaf node
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

def get_feature_count(tree) -> int:
    """Get the number of features used by the tree."""
    return tree.tree_.n_features

def main():
    parser = argparse.ArgumentParser(description='Convert VIPER decision tree to minimal Python code (Ultra Fast Version)')
    parser.add_argument('-i', '--input', help='Input directory containing VIPER model files', required=True)
    parser.add_argument('-n', '--name', help='Input file name (optional, will use the one with most leaves if not specified)')
    parser.add_argument('-o', '--output', help='Output file name (optional, will be generated based on number of leaves)')
    parser.add_argument('-g', '--game', help='Game name', required=True)
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Load the model
    model, num_leaves = load_model(args.input, args.name)
    
    # Get number of features for debugging
    feature_count = get_feature_count(model)
    if args.debug:
        print(f"This model uses {feature_count} features")
        print("Warning: This fast version assumes that all features (including combined features) are directly available in the state vector")
        print("No feature calculation is performed, so make sure your environment provides all needed features")
    
    # Convert tree to code
    code = tree_to_code_fast(model)
    
    # Create output directory if it doesn't exist
    input_dir_name = os.path.basename(os.path.normpath(args.input))
    output_dir = os.path.join("resources", "program_policies_viper", input_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file name
    if args.output:
        output_file = os.path.join(output_dir, args.output)
    else:
        output_file = os.path.join(output_dir, f"play_viper_fast_{num_leaves}_leaves.py")
    
    # Write code to file
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"Generated ultra-fast code has been saved to {output_file}")
    print(f"IMPORTANT: This code assumes that the environment directly provides all {feature_count} features in the state vector")

if __name__ == "__main__":
    main() 