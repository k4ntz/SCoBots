from termcolor import colored
# TODO: make nonstatic
OCATARI_AVAILABLE_GAMES = ["Boxing", "Breakout", "Skiing", "Pong", "Seaquest", "Tennis"]



# file to delegate to different object producing environments (wrappers)
def make(env_name, notify=False):
    if notify:
        print("Env Name:", env_name)
    if True: # check if game is available and delegate
        import scobi.environments.ocgym as ocgym
        env = ocgym.make(env_name, notify)
        # TODO: get env name from OC_atari instance
        print(colored("scobi >", "blue"), "Environment %s specified. Compatibale object extractor %s loaded." % (colored(env_name, "yellow"),colored("OC_Atari", "yellow")))
        return env