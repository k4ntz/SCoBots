# file to load the correct env

# notify: Print creations
def make(cfg, notify=False):
    env_name = cfg.env_name
    if notify:
        print("Env Name:", env_name)
    if "coin" in env_name:
        import xrl.environments.pgym as pgym
        return pgym.make(env_name, notify)
    else:
        import xrl.environments.agym as agym
        return agym.make(env_name, notify)
