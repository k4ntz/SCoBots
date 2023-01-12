from termcolor import colored
# TODO: make nonstatic
OCATARI_AVAILABLE_GAMES = ["Boxing", "Breakout", "Skiing", "Pong", "Seaquest", "Tennis"]



# file to delegate to different object producing environments (wrappers)
def make(env_name, logger, notify=False):
    if notify:
        print("Env Name:", env_name)
    if True: # check if game is available and delegate
        import scobi.environments.ocgym as ocgym
        env = ocgym.make(env_name, notify)
        # TODO: get env name from OC_atari instance
        logger.GeneralInfo("Environment %s specified. Compatible object extractor %s loaded." % (colored(env_name, "light_cyan"),colored("OC_Atari", "light_cyan")))
        return env