import os
from huggingface_hub import Repository

hf_token = os.getenv('HF_TOKEN')
repository = os.getenv('HF_REPO_ID')

directory = 'resources/con'

repository = Repository(local_dir=directory, clone_from=repository, use_auth_token=hf_token)

repository.git_add(auto_lfs_track=True)
repository.git_commit('Uploading SCoBots checkpoints')

repository.git_push()

print("Uploaded checkpoints successfully")