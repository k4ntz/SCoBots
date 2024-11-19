import os

import yaml


class ModelCard:
    def __init__(self, game="", envs="", obs="", prune="", seed="", reward=""):
        self.game = game
        self.num_timesteps = ""
        self.total_timesteps = ""
        self.sde_sample_freq = ""
        self.n_envs = envs
        self.n_epochs = ""
        self.gae_lambda = ""
        self.n_steps = ""
        self.batch_size = ""
        self.ent_coef = ""
        self.gamma = ""
        self.policy = ""
        self.obs_type = obs
        self.pruned = prune
        self.seed = seed
        self.reward_func = reward

    def load_card(self, path):
        try:
            with open(path, "r") as f:
                # Split YAML metadata and Markdown content
                parts = f.read().split("---", 2)
                if len(parts) >= 3:
                    metadata = yaml.safe_load(parts[1]) or {}
                    content = parts[2].strip()
                    return metadata, content
                else:
                    print(f"Invalid model card format in {path}.")
        except Exception as e:
            print(f"Error reading {path}: {e}")

    def create_card(self, path):
        yaml_metadata = (
                f"---\n"
                f"language: en\n"
                f"tags:\n"
                f"- stable-baselines3\n"
                f"- reinforcement-learning\n"
                f"- {self.game.lower()}\n"
                f"- {self.obs_type.lower()}\n"
                f"model:\n"
                f"  seed: {self.seed}\n"
                f"  reward_function: {self.reward_func}\n"
                f"  observation_type: {self.obs_type}\n"
                f"  pruned: {self.pruned}\n"
                f"  normalized: Yes\n"
                f"  num_timesteps: {self.num_timesteps}\n"
                f"  total_timesteps: {self.total_timesteps}\n"
                f"  sde_sample_freq: {self.sde_sample_freq}\n"
                f"  n_envs: {self.n_envs}\n"
                f"  n_epochs: {self.n_epochs}\n"
                f"  gae_lambda: {self.gae_lambda}\n"
                f"  n_steps: {self.n_steps}\n"
                f"  batch_size: {self.batch_size}\n"
                f"  gamma: {self.gamma}\n"
                f"  ent_coef: {self.ent_coef}\n"
                f"license: mit\n"
                f"---\n"
            )

        readme_content = (
                f"# Model Card for {self.game}\n\n"
                f"## Overview\n"
                f"This model is trained for the **{self.game}** game. It was trained using the following configurations:\n\n"
                f"- **Seed**: {self.seed}\n"
                f"- **Reward Function**: {self.reward_func}\n"
                f"- **Observation Type**: {self.obs_type}\n"
                f"- **Pruned**: {self.pruned}\n\n"
                f"## Training Details\n"
                f"- **Framework**: Stable-Baselines3 (SB3)\n"
                f"- **Policy**: {self.policy}\n"
                f"- **Number of Timesteps**: {self.num_timesteps}\n"
                f"- **Total Timesteps**: {self.total_timesteps}\n"
                f"- **SDE Sample Frequency**: {self.sde_sample_freq}\n"
                f"- **Number of Environments**: {self.n_envs}\n"
                f"- **Number of Epochs**: {self.n_epochs}\n"
                f"- **GAE Lambda**: {self.gae_lambda}\n"
                f"- **Number of Steps**: {self.n_steps}\n"
                f"- **Batch Size**: {self.batch_size}\n"
                f"- **Gamma**: {self.gamma}\n"
                f"- **Ent_coef**: {self.ent_coef}\n"
                f"- **Reward Details**: {self.reward_func}\n\n"
                f"## Usage\n"
                f"For more detailed usage, visit https://github.com/k4ntz/SCoBots and its README which explains in detail how to use SCoBots\n\n"
                f"## License\n"
                f"This model is released under the MIT License. See the LICENSE file for more details.\n"
            )

        full_readme = yaml_metadata + readme_content

        readme_path = os.path.join(path, "README.md")
        with open(readme_path, "w") as f:
            f.write(full_readme)

    def update_card(self, path, num_steps="", total_steps="", sde_freq="", epochs="", lamba="",
                    n_steps="", batch="", coef="", gamma="", policy=""):
        self.num_timesteps = num_steps
        self.total_timesteps = total_steps
        self.sde_sample_freq = sde_freq
        self.n_epochs = epochs
        self.gae_lambda = lamba
        self.n_steps = n_steps
        self.batch_size = batch
        self.ent_coef = coef
        self.gamma = gamma
        self.policy = policy
        self.create_card(path)



