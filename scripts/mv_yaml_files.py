import shutil
from pathlib import Path

def extract_game_name(folder_name: str) -> str:
    # strip it bby
    return folder_name.split("_")[0]

def process_checkpoints(checkpoints_dir: Path, focusfiles_dir: Path):
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"WHERE CHECKPOINT AT")
    if not focusfiles_dir.is_dir():
        raise FileNotFoundError(f"WHEREFOCUSFILES AT")

    for folder in checkpoints_dir.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            game_name = extract_game_name(folder_name)  # Extract game name
            if folder_name.endswith("pruned"):
                yaml_file_name = f"pruned_{game_name.lower()}.yaml"
            else:
                yaml_file_name = f"default_focus_{game_name.capitalize()}-v5.yaml"

            source_file = focusfiles_dir / yaml_file_name
            if not source_file.exists():
                print(f"Warning: Source file not found: {source_file}")
                continue

            # final destination 4/10
            destination_file = folder / yaml_file_name

            # Copy copy copy copy copy copy copy copy
            shutil.copy(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")

checkpoints_path = Path("checkpoints")
focusfiles_path = Path("focusfiles")

# Run it DMC
process_checkpoints(checkpoints_path, focusfiles_path)
