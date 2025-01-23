from sklearn.tree import _tree
import numpy as np
import argparse
from joblib import load
from pathlib import Path
import os


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

def load_interpreter_tree(folder_name, name=None):
    # load file with the most leaves
    tree_files = [f for f in os.listdir(folder_name) if f.endswith(".pkl")]
    if name is None:
        try:
            tree_files.sort(key=lambda x: int(x.split("leaves")[0].split("tree")[-1]))
            tree = load(folder_name + "/" + tree_files[-1])
        except:
            #choose the lastest file
            tree_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_name, x)))
            tree = load(folder_name + "/" + tree_files[-1])
        return tree
    else:
        tree = load(folder_name + "/" + name)
        return tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="folder name containing '.pkl'")
    parser.add_argument("-n", "--name", type=str, required=False, help="name of the input file")
    parser.add_argument("-o", "--output", type=str, required=False, help="output file name")

    opts = parser.parse_args()
    tree = load_interpreter_tree(opts.input, opts.name)
    name = opts.input.split("/")[-1]
    if opts.output is None:
        file_name = "play_python_"+ str(tree.get_n_leaves()) + "_leaves.py"
        opts.output = file_name

    output_file_path = Path("resources/program_policies", name, opts.output)

    # Generate Python code for decision tree
    tree_code = tree_to_code(tree)

    os.makedirs(output_file_path.parent, exist_ok=True)

    # Save generated code to file
    with open(output_file_path, "w") as f:
        f.write("def play(state):\n")
        f.write(tree_code)
        f.write("\n    return -1  # default return -1")

    print("Generated code saved to " + str(output_file_path))

if __name__ == '__main__':
    main()