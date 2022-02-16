# file to load the correct env

import xrl.environments.agym as agym
import xrl.environments.pgym as pgym

def make(cfg):
    env_name = cfg.env_name
    print("Env Name:", env_name)
    if "coin" in env_name:
        return pgym.make(env_name)
    else:
        return agym.make(env_name)
