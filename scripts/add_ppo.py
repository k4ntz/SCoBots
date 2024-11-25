import os

# Root directory containing subdirectories with README.md files
root_directory = "resources/checkpoints"

# Line to insert
new_line = "  rl_algorithm: PPO\n"

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename == "README.md":
            print(f"Updating {os.path.join(dirpath, filename)}")
            file_path = os.path.join(dirpath, filename)
            
            # Read the file
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            # Find the line starting with "model:" and insert the new line after it
            updated_lines = []
            for line in lines:
                updated_lines.append(line)
                if line.strip().startswith("model:"):
                    updated_lines.append(new_line)
            # Write the updated content back to the file
            with open(file_path, "w") as file:
                file.writelines(updated_lines)

print("Updated all README.md files in subdirectories.")