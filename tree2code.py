from sklearn.tree import _tree
import numpy as np
import argparse
from joblib import load
from pathlib import Path
import os
from scobi import Environment
from utils.feature_utils import mask_features, auto_generate_mask
import yaml


def tree_to_code(tree):
    """
    Convert decision tree to Python if-else logic code.
    :param tree: Decision tree object
    :return: Python logic code string
    """
    tree_ = tree.tree_
    output = []

    def recurse(node, depth):
        """
        Recursively generate Python code for decision tree nodes.
        """
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_index = tree_.feature[node]  # Use feature index
            threshold = tree_.threshold[node]
            output.append(f"{indent}if state[{feature_index}] <= {threshold:.6f}:")
            recurse(tree_.children_left[node], depth + 1)
            output.append(f"{indent}else:")
            recurse(tree_.children_right[node], depth + 1)
        else:
            action = np.argmax(tree_.value[node])
            output.append(f"{indent}return {action}")

    recurse(0, 1)
    return "\n".join(output)

def get_oblique_data_strings(features):
    """
    Simulate get_oblique_data processing on a list of strings,
    generating a list of strings (original features + pairwise differences) with the same subtraction direction as the numerical version.
    
    In the numerical version, diffs[b, i, j] = S[b, j] - S[b, i].
    Here, we need to generate (features[j] - features[i]).
    """
    n_features = len(features)
    i_lower, j_lower = np.tril_indices(n_features, k=-1)
    
    diffs = []
    for i, j in zip(i_lower, j_lower):
        diffs.append(f"{features[j]} - {features[i]}")

    final_features = features + diffs
    return final_features

def replace_feature_names(code, feature_names):
    """
    Replace feature indices with feature names in the generated code.
    """
    for i, feature_name in enumerate(feature_names):
        code = code.replace(f"state[{i}]", feature_name)
    return code

def get_feature_names(env, ff_file_path, pruned_ff_name):
    """
    get the feature names of the environment
    
    Parameters
    ----------
    env : Environment
        the environment object
    ff_file_path : str
        the path of the feature filter configuration file
    pruned_ff_name : str
        the name of the feature filter configuration file
    
    Returns
    -------
    list
        the filtered feature names
    """
    try:
        # get the original feature descriptions
        feature_descriptions = env.get_vector_entry_descriptions()
        
        # try to get the mask_indices from the configuration file
        mask_indices = None
        try:
            with open(os.path.join(ff_file_path, pruned_ff_name), 'r') as f:
                config = yaml.safe_load(f)
            mask_indices = config.get('FEATURE_MASK', {}).get('keep_indices', None)
        except (FileNotFoundError, yaml.YAMLError):
            pass
            
        # if the configuration file does not exist or the mask_indices is not found, auto generate the mask_indices
        if mask_indices is None:
            mask_indices = auto_generate_mask(feature_descriptions)
            
        # apply the mask to get the filtered feature names
        feature_names = mask_features(feature_descriptions, mask_indices)
        
        # replace the time suffix with _prev
        feature_names = [name.replace("[t-1]", "_prev") for name in feature_names]
        return feature_names
        
    except AttributeError:
        print("Warning: Environment does not support get_vector_entry_descriptions")
        return []

def load_interpreter_tree(folder_name, name=None):
    # load the file with the most leaves
    tree_files = [f for f in os.listdir(folder_name) if f.endswith(".pkl")]
    print(tree_files)
    if name is None:
        try:
            tree_files.sort(key=lambda x: int(x.split("leaves")[0].split("tree")[-1]))
            tree = load(folder_name + "/" + tree_files[-1])
        except:
            # choose the latest file
            tree_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_name, x)))
            tree = load(folder_name + "/" + tree_files[-1])
        return tree
    else:
        tree = load(folder_name + "/" + name)
        return tree
    
def generate_variable_mapping(feature_names):
    """
    Generate class definitions and a mapping of feature names to their corresponding indices.
    """
    # extract class names from feature names
    class_definitions = set()
    for feature_name in feature_names:
        class_name = feature_name.split('.')[0]
        class_definitions.add(class_name)

    class_definitions_code = "\n".join([f"    class {class_name}:\n        pass" for class_name in class_definitions])

    mapping_code = ["    # Mapping of feature names to their corresponding indices"]
    for i, feature_name in enumerate(feature_names):
        variable_name = feature_name.replace('[t-1]', '_prev')
        mapping_code.append(f"    {variable_name} = state[{i}]")
    return class_definitions_code + "\n" + "\n".join(mapping_code)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="folder name containing '.pkl'")
    parser.add_argument("-n", "--name", type=str, required=False, help="name of the input file")
    parser.add_argument("-o", "--output", type=str, required=False, help="output file name")
    parser.add_argument("-g", "--game", type=str, required=False, help="game name")
    parser.add_argument("-ff", "--focus_file", type=str, required=False, help="focus file")
    
    opts = parser.parse_args()
    tree = load_interpreter_tree(opts.input, opts.name)
    name = opts.input.split("/")[-1]
    if opts.output is None:
        file_name = "play_python_"+ str(tree.get_n_leaves()) + "_leaves.py"
        opts.output = file_name

    output_file_path = Path("resources/program_policies", name, opts.output)

    env_str = "ALE/" + opts.game + "-v5"
    focus_dir = opts.focus_file.rsplit("/", 1)[0]
    pruned_ff_name = opts.focus_file.split("/")[-1]

    env = Environment(env_str,
                    focus_dir=focus_dir,
                    focus_file=pruned_ff_name,
                    hide_properties=False,
                    draw_features=True,
                    reward=0)

    # Generate Python code for decision tree
    tree_code = tree_to_code(tree)

    orignal_features = get_feature_names(env, focus_dir, pruned_ff_name)
    feature_names = get_oblique_data_strings(orignal_features)
    tree_code = replace_feature_names(tree_code, feature_names)
    variable_mapping_code = generate_variable_mapping(orignal_features)

    os.makedirs(output_file_path.parent, exist_ok=True)

    # Save generated code to file
    with open(output_file_path, "w") as f:
        f.write("def play(state):\n")
        f.write(variable_mapping_code + "\n")
        f.write(tree_code)
        f.write("\n    return -1  # default return -1")

    print("Generated code saved to " + str(output_file_path))

if __name__ == '__main__':
    main()