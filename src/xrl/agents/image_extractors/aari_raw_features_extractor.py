# "image" extractor for atariari
# in reality just a wrapper but necessary for agent-object


# wrapper for atariari to get labels
def get_labels(obs, env_info, gametype=0):
    labels = env_info
    # extract raw features
    if gametype != 3:
        #labels = env_info["labels"]
        labels = env_info["objects"]
    return labels