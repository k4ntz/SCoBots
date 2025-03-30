"""
convert VIPER decision tree to Python code
convert VIPER learned decision tree model to pure Python code, supporting feature combinations
"""

import argparse
import math
import numpy as np
from joblib import load
from pathlib import Path
import os
from sklearn.tree import _tree
from scobi import Environment
import re

# define a small constant, used to avoid division by zero error, same as concepts.py
EPS = np.finfo(np.float64).eps.item()

def sanitize_var_name(feature_name):
    """
    convert feature name to a valid Python variable name
    
    parameters:
    feature_name: feature name
    
    return:
    valid Python variable name
    """
    # replace special characters
    var_name = feature_name
    # first replace the dot and time suffix
    var_name = var_name.replace(".", "_").replace("[t-1]", "_prev")
    
    # then process the parentheses and other special characters
    if "(" in var_name:
        # for features like "D(Player1, Ball1).x", convert to "D_Player1_Ball1_x"
        var_name = var_name.replace("(", "_").replace(")", "").replace(", ", "_").replace(",", "_")
        var_name = var_name.replace(" ", "")
    
    return var_name


def process_combined_features(feature_names):
    """
    process the feature list, distinguish between original features and combined features, inline the calculation logic directly into the code
    
    parameters:
    feature_names: feature description list
    
    return:
    (original feature list, combined feature dictionary {name: calculation code})
    """
    # distinguish between original features and combined features
    original_features = []
    combined_features = {}
    
    for feature in feature_names:
        if "(" in feature:  # combined features
            # generate combined feature code based on feature type
            if feature.startswith("D("):
                # distance feature: D(obj1, obj2).axis
                match = re.match(r"D\(([^,]+),\s*([^\)]+)\)\.([xy])", feature)
                if match:
                    obj1, obj2, axis = match.groups()
                    obj1, obj2 = obj1.strip(), obj2.strip()
                    
                    # inline distance calculation logic
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
                    
                    # inline Euclidean distance calculation logic
                    combined_features[feature] = f"math.sqrt(({obj2}_y - {obj1}_y)**2 + ({obj2}_x - {obj1}_x)**2)"
            
            elif feature.startswith("C("):
                # center point: C(obj1, obj2).axis
                match = re.match(r"C\(([^,]+),\s*([^\)]+)\)\.([xy])", feature)
                if match:
                    obj1, obj2, axis = match.groups()
                    obj1, obj2 = obj1.strip(), obj2.strip()
                    
                    # inline center point calculation logic
                    if axis == 'x':
                        combined_features[feature] = f"({obj1}_x + {obj2}_x) / 2"
                    else:  # axis == 'y'
                        combined_features[feature] = f"({obj1}_y + {obj2}_y) / 2"
            
            elif feature.startswith("V("):
                # velocity: V(obj).axis
                match = re.match(r"V\(([^\)]+)\)\.([xy])", feature)
                if match:
                    obj, axis = match.groups()
                    obj = obj.strip()
                    
                    # inline velocity calculation logic
                    combined_features[feature] = f"math.sqrt(({obj}_x_prev - {obj}_x)**2 + ({obj}_y_prev - {obj}_y)**2)"
            
            elif feature.startswith("DV("):
                # direction velocity: DV(obj).axis
                match = re.match(r"DV\(([^\)]+)\)\.([xy])", feature)
                if match:
                    obj, axis = match.groups()
                    obj = obj.strip()
                    
                    # inline direction velocity calculation logic
                    if axis == 'x':
                        combined_features[feature] = f"{obj}_x_prev - {obj}_x"
                    else:  # axis == 'y'
                        combined_features[feature] = f"{obj}_y_prev - {obj}_y"
            
            elif feature.startswith("LT("):
                # linear trajectory: LT(obj1, obj2).axis
                match = re.match(r"LT\(([^,]+),\s*([^\)]+)\)\.([xy])", feature)
                if match:
                    obj1, obj2, axis = match.groups()
                    obj1, obj2 = obj1.strip(), obj2.strip()
                    
                    # inline linear trajectory calculation logic
                    # first calculate the slope and intercept
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


def tree_to_code(tree, feature_names, combined_features=None):
    """
    convert the decision tree to Python if-else logic code.
    
    parameters:
    tree: decision tree object
    feature_names: feature name list
    combined_features: combined feature dictionary {feature name: calculation code}
    
    return:
    Python logic code string
    """
    combined_features = combined_features or {}
    tree_ = tree.tree_
    output = []
    
    # check if the number of features matches
    n_features = tree_.n_features
    if len(feature_names) != n_features:
        print(f"warning: feature name number ({len(feature_names)}) does not match the decision tree feature number ({n_features})")
        if len(feature_names) < n_features:
            # use default feature names to fill the missing part
            feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), n_features)]
    
    def recurse(node, depth):
        """
        recursively generate the Python code for the decision tree node.
        """
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_index = tree_.feature[node]
            threshold = tree_.threshold[node]
            
            if feature_index >= len(feature_names):
                print(f"warning: feature index {feature_index} exceeds the feature name list range {len(feature_names)}")
                feature_code = f"state[{feature_index}]"
                output.append(f"{indent}if {feature_code} <= {threshold:.6f}:")
            else:
                # get the feature name
                feature_name = feature_names[feature_index]
                
                # if it is a combined feature, use its calculation code
                if feature_name in combined_features:
                    feature_code = combined_features[feature_name]
                    output.append(f"{indent}if {feature_code} <= {threshold:.6f}:")
                else:
                    # original feature, use the variable name directly
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


def load_interpreter_tree(folder_name, name=None):
    """
    load the decision tree model
    
    parameters:
    folder_name: model file folder path
    name: model file name, if None, select the tree with the most leaves
    
    return:
    decision tree model
    """
    # load the file with the most leaves
    tree_files = [f for f in os.listdir(folder_name) if f.endswith(".pkl") or f.endswith(".viper")]
    if name is None:
        try:
            # sort by the number of leaves
            tree_files.sort(key=lambda x: int(x.split("leaves")[0].split("tree")[-1]) if "leaves" in x else 0)
            tree = load(os.path.join(folder_name, tree_files[-1]))
        except:
            # select the latest file
            tree_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_name, x)))
            tree = load(os.path.join(folder_name, tree_files[-1]))
        return tree
    else:
        tree = load(os.path.join(folder_name, name))
        return tree


def generate_feature_extraction_code(feature_names):
    """
    generate feature extraction code, process duplicate and special characters
    
    parameters:
    feature_names: feature name list
    
    return:
    feature extraction code string
    """
    code = ["    # extract original features from the state vector"]
    var_names = {}
    
    for i, feature_name in enumerate(feature_names):
        # generate a valid variable name
        var_name = sanitize_var_name(feature_name)
        
        # process duplicate variable names (only keep the first occurrence)
        if var_name in var_names:
            continue
        
        var_names[var_name] = i
        code.append(f"    {var_name} = state[{i}]")
    
    return "\n".join(code)


def main():
    parser = argparse.ArgumentParser(description="convert VIPER decision tree to pure Python code")
    parser.add_argument("-i", "--input", type=str, required=True, help="folder name containing '.pkl' or '.viper' files")
    parser.add_argument("-n", "--name", type=str, required=False, help="input file name")
    parser.add_argument("-o", "--output", type=str, required=False, help="output file name")
    parser.add_argument("-g", "--game", type=str, required=True, help="game name")
    parser.add_argument("-ff", "--focus_file", type=str, required=True, help="focus file")
    parser.add_argument("--debug", action="store_true", help="output debug information")
    
    opts = parser.parse_args()
    tree = load_interpreter_tree(opts.input, opts.name)
    print(f"decision tree feature number: {tree.tree_.n_features}")
    name = opts.input.split("/")[-1]
    
    if opts.output is None:
        file_name = "play_viper_python_"+ str(tree.get_n_leaves()) + "_leaves.py"
        opts.output = file_name

    output_dir = Path("resources/program_policies_viper", name)
    output_file_path = output_dir / opts.output

    env_str = "ALE/" + opts.game + "-v5"
    focus_dir = opts.focus_file.rsplit("/", 1)[0]
    pruned_ff_name = opts.focus_file.split("/")[-1]
    focus_file_path = os.path.join(focus_dir, pruned_ff_name)

    # create environment instance
    env = Environment(env_str,
                      focus_dir=focus_dir,
                      focus_file=pruned_ff_name,
                      hide_properties=False,
                      draw_features=True,
                      reward=0)
    
    # get the environment feature dimension
    try:
        obs_shape = env.observation_space.shape
        print(f"environment observation space shape: {obs_shape}")
    except Exception as e:
        print(f"failed to get environment observation space shape: {e}")
    
    # get the feature list from the environment
    feature_names = []
    try:
        feature_names = env.get_vector_entry_descriptions()
        print(f"got {len(feature_names)} features from the environment")
        
        if opts.debug and feature_names:
            print("environment feature list:")
            for i, feat in enumerate(feature_names):
                print(f"  {i}: {feat}")
    except Exception as e:
        print(f"failed to get feature descriptions from the environment: {e}")
        print(f"warning: no available feature names, but the decision tree needs {tree.tree_.n_features} features")
        print("the code may not be generated correctly. Please ensure the environment returns the correct feature descriptions.")
    
    # check if the feature number is enough
    if not feature_names:
        print("error: failed to get feature names, cannot generate correct code")
        return
    
    if len(feature_names) != tree.tree_.n_features:
        print(f"warning: environment feature number ({len(feature_names)}) does not match the decision tree feature number ({tree.tree_.n_features})")
        print("the generated code may be incorrect. Please check the environment settings")
        
        # fill or truncate the feature list to match the tree feature number
        if len(feature_names) < tree.tree_.n_features:
            # use generic names to fill the missing part
            feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), tree.tree_.n_features)]
            print(f"filled the feature list with generic names to {len(feature_names)} features")
        else:
            # truncate the extra features
            feature_names = feature_names[:tree.tree_.n_features]
            print(f"truncated the feature list to {len(feature_names)} features")
    
    # process the combined feature calculation logic
    _, combined_features = process_combined_features(feature_names)
    
    if opts.debug:
        print("combined feature calculation:")
        for feat, code in combined_features.items():
            print(f"  {feat}: {code}")

    # generate Python code
    tree_code = tree_to_code(tree, feature_names, combined_features)
    feature_extraction_code = generate_feature_extraction_code(feature_names)

    os.makedirs(output_file_path.parent, exist_ok=True)

    # import necessary libraries
    import_code = (
        "import numpy as np\n"
        "import math\n\n"
        "# define a small constant, used to avoid division by zero error\n"
        "EPS = np.finfo(np.float64).eps.item()\n\n"
    )

    # save the generated code to the file
    with open(output_file_path, "w") as f:
        f.write(import_code)
        f.write("def play(state):\n")
        f.write(feature_extraction_code + "\n\n")
        f.write(tree_code)
        f.write("\n    return -1  # default return -1")

    print(f"generated code saved to {output_file_path}")


if __name__ == '__main__':
    main() 