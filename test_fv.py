from scobi import Environment
import numpy as np
# Extensive init, interactive, custom fcous dir and focus file
env = Environment(env_name='PongDeterministic-v4', interactive=True, focus_dir="experiments/my_focusfiles", focus_file="pruned_pong.yaml")
env.reset()
obs, env_reward, scobi_reward, truncated, terminated, info, obs_a = env.step(1)
env.close()

fv_desc = env.feature_vector_description
fv_strs = fv_desc[0]
fv_backmap = fv_desc[1]


def format_feature(feature_name, feature_signature, ii):
    if feature_name == 'RGB':
        axis = ["R", "G", "B"][ii]
        return f"RGB({feature_signature}.{axis})"
    if feature_name == "POSITION_HISTORY":
        if ii < 2:
            axis = ["x", "y"][ii]
            return f"{feature_signature}.{axis}"
        axis = ["x", "y"][ii-2]
        return f"{feature_signature}.{axis}[t-1]"
    axis = ["x", "y"][ii]
    if ii > 3:
        print("error")
        exit()
    if feature_name == 'POSITION':
        return f"{feature_signature}.{axis}"
    elif feature_name == "EUCLIDEAN_DISTANCE":
        return f"ED({feature_signature})"
    elif feature_name == "DISTANCE":
        return f"D({feature_signature[0][1]},{feature_signature[1][1]}).{axis}"
    elif feature_name == "VELOCITY":
        return f"V({feature_signature[0][1]}).{axis}"
    elif feature_name == "DIR_VELOCITY":
        return f"DV({feature_signature[0][1]}).{axis}"
    elif feature_name == "CENTER":
        return f"C({feature_signature[0][1]},{feature_signature[1][1]}).{axis}"
    elif feature_name == "ORIENTATION":
        return f"O({feature_signature})"
    elif feature_name == "LINEAR_TRAJECTORY":
        return f"LT({feature_signature[0][1]},{feature_signature[1][1]}).{axis}"
    elif feature_name == "COLOR":
        return f"COL({feature_signature})"


desc_strs = []
for i, f in enumerate(fv_strs):
    feature_name = f[0]
    feature_signature = f[1]
    idxs = np.where(fv_backmap == i)[0]
    for ii in range(len(idxs)):
        desc_strs.append(format_feature(feature_name, feature_signature, ii))


print(desc_strs)
print(len(desc_strs))