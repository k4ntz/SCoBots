# file to load the correct env

import xrl.environments.agym as agym
import xrl.environments.pgym as pgym

# notify: Print creations
def make(cfg, notify=False):
    env_name = cfg.env_name
    if notify:
        print("Env Name:", env_name)
    if "coin" in env_name:
        return pgym.make(env_name, notify)
    else:
        return agym.make(env_name, notify)
