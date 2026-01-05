import os
import sys
import glob
import re
import joblib

# Get the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
from scobi import Environment

base_folder_path = os.path.join(PROJECT_ROOT, 'resources/viper_extracts/extract_output')
file_pattern = os.path.join(base_folder_path, '**', 'Tree-*best.viper')
matching_files = glob.glob(file_pattern, recursive=True)

def get_feature_names_from_scobi(env_name, focus_file_path):
    """Get feature names from SCOBI environment"""
    env = Environment(
        env_name,
        focus_dir="",
        focus_file=focus_file_path,
        hide_properties=False,
        draw_features=True,
        reward=0
    )
    env.reset()
    feature_names = env.get_vector_entry_descriptions()
    env.close()
    return feature_names

def find_focus_file(folder_path):
    """Find a yaml file that contains 'v5' or starts with 'pruned' in the given folder"""
    yaml_files = glob.glob(os.path.join(folder_path, '*.yaml'))
    yaml_files += glob.glob(os.path.join(folder_path, '*.yml'))

    for yaml_file in yaml_files:
        filename = os.path.basename(yaml_file)
        if 'v5' in filename or filename.startswith('pruned'):
            return yaml_file

    return None

# Extract the if-else from tree
def extract_tree_body_as_code(tree, feature_names, class_names, node=0, depth=0):
    # If leaf node
    if tree.tree_.feature[node] == -2:
        class_label_str = class_names[tree.tree_.value[node].argmax()]
        class_label_int = int(class_label_str)
        return f"{'    ' * depth}return {class_label_int}\n"

    feature_index = tree.tree_.feature[node]
    threshold = tree.tree_.threshold[node]

    # Use actual feature name directly
    feature_name = feature_names[feature_index]
    condition = f"{feature_name} <= {threshold:.2f}"

    left_code = extract_tree_body_as_code(
        tree, feature_names, class_names,
        tree.tree_.children_left[node],
        depth + 1
    )
    right_code = extract_tree_body_as_code(
        tree, feature_names, class_names,
        tree.tree_.children_right[node],
        depth + 1
    )

    code = (
        f"{'    ' * depth}if {condition}:\n"
        f"{left_code}"
        f"{'    ' * depth}else:\n"
        f"{right_code}"
    )
    return code


# Main loop over files
for viper_path in matching_files:
    folder_dir = os.path.dirname(viper_path)
    folder_name = os.path.basename(folder_dir)
    folder_name_no_extraction = folder_name.replace("-extraction", "")

    # Look for focus file in resources/checkpoints/
    checkpoint_dir = os.path.join(PROJECT_ROOT, "resources", "checkpoints", folder_name_no_extraction)

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}, skipping.")
        continue

    focus_full_path = find_focus_file(checkpoint_dir)
    if focus_full_path is None:
        print(f"No suitable focus file found in {checkpoint_dir}, skipping.")
        continue

    target_dir = checkpoint_dir  # Write output to the same checkpoint directory
    os.makedirs(target_dir, exist_ok=True)

    script_name = f"{folder_name_no_extraction}-tree-rules.py"
    script_path = os.path.join(target_dir, script_name)

    if not os.path.exists(viper_path):
        print(f"No Tree-*_best.viper files found in {folder_dir}, skipping.")
        continue

    # get the environment name from the folder_name
    words = folder_name_no_extraction.split("_")
    game_name = words[0]
    env_name = f"ALE/{game_name}-v5"

    # Load the DecisionTree and extract code
    dtree = joblib.load(viper_path)

    features = get_feature_names_from_scobi(env_name, focus_full_path)

    tree_body_code = extract_tree_body_as_code(
        dtree,
        features,
        dtree.classes_,
        node=0,
        depth=0
    )

    # Write tree
    with open(script_path, "w") as f:
        f.write(tree_body_code)

    print(f"Generated Python script at: {script_path}")
