from sklearn.tree import _tree
import numpy as np
import argparse
from joblib import load
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

def load_interpreter_tree(folder_name):
    tree = load(folder_name + "/tree.pkl")
    return tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="folder name containing 'tree.pkl'")
    parser.add_argument("-o", "--output", type=str, required=False, default="tree_play.py", help="output file name")
    opts = parser.parse_args()
    tree = load_interpreter_tree(opts.input)

    # Generate Python code for decision tree
    tree_code = tree_to_code(tree)

    # Save generated code to file
    with open(opts.output, "w") as f:
        f.write("def play(state):\n")
        f.write(tree_code)
        f.write("\n    return -1  # default return -1")

if __name__ == '__main__':
    main()