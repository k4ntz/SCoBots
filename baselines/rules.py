USE_REMIX = True
if USE_REMIX:
    from remix import eclaire
    import tensorflow as tf
    from tensorflow import keras
    #from tensorflow.keras.models import Dense, InputLayer
else:
    from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor
    from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
    import gymnasium as gym

import os
import torch
import numpy as np
#from fnames import get_actions_for_game, get_fnames_for_game
from scobi.focus import Focus
from eclaire_utils.utils import get_eclaire_config



eclaire_cfg = get_eclaire_config()

focus_filename = eclaire_cfg.focus_filename
focus_dir = eclaire_cfg.focus_dir
focus_object = Focus(fofiles_dir_name=focus_dir, fofile=focus_filename)
actions = focus_object.PARSED_ACTIONS
fnames = focus_object.get_vector_entry_descriptions()
fnames = [f.replace(" ", "") for f in fnames]
eclaire_dir = eclaire_cfg.eclaire_dir

input_size = len(fnames)
output_size = len(actions)

# load weights
checkpoint = torch.load(os.path.join(eclaire_dir, eclaire_cfg.model_filename)) # was "checkpoint.pth")
# create test input vector
np.random.seed(2024) # set seed for reproducibility
input_vector = torch.tensor(np.random.rand(1, input_size).astype(np.float32))


if not USE_REMIX:
    # load pytorch ppo model for comparison
    if eclaire_cfg.num_layers == 1:
        net_arch = dict(pi=[64], vf=[64])
    elif eclaire_cfg.num_layers == 2:
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    else:
        raise ValueError("eclaire_cfg.num_layers must be 1 or 2")
    activation_fn = torch.nn.ReLU
    pkwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(input_size,), dtype=np.float32)
    action_space = gym.spaces.Discrete(6)
    policy_net = ActorCriticPolicy(
        observation_space,
        action_space,
        BasePolicy._dummy_schedule,
        net_arch= pkwargs["net_arch"],
        activation_fn=pkwargs["activation_fn"],
    )
    policy_net.load_state_dict(checkpoint)

    policy_net.eval()
    features = policy_net.mlp_extractor.forward_actor(input_vector)
    logits = policy_net.action_net.forward(features)
    probs = torch.nn.functional.softmax(logits, dim=1)
    print("probs: ", probs)
    print("logits: ", logits)
    print("actions: ", logits.argmax(dim=1))
    #actions, values, log_prob = policy_net.forward(input_vector, deterministic=True)
    #print("actions: ", actions)
    #print("values: ", values)
    #print("log_prob: ", log_prob)
else:
    if eclaire_cfg.num_layers == 1:
        weights = [checkpoint['mlp_extractor.policy_net.0.weight'], 
               checkpoint['mlp_extractor.policy_net.0.bias'],
               #checkpoint['mlp_extractor.policy_net.2.weight'],
               #checkpoint['mlp_extractor.policy_net.2.bias'],
               checkpoint['action_net.weight'],
               checkpoint['action_net.bias'],]
    elif eclaire_cfg.num_layers == 2:
        weights = [checkpoint['mlp_extractor.policy_net.0.weight'], 
               checkpoint['mlp_extractor.policy_net.0.bias'],
               checkpoint['mlp_extractor.policy_net.2.weight'],
               checkpoint['mlp_extractor.policy_net.2.bias'],
               checkpoint['action_net.weight'],
               checkpoint['action_net.bias'],]
    else:
        raise ValueError("eclaire_cfg.num_layers must be 1 or 2")
    
    weights = [w.cpu().numpy().T for w in weights]      

    hidden_size = 64
    act_f = "relu"

    # define keras model
    keras_model = keras.Sequential()
    keras_model.add(keras.Input(shape=(input_size,)))
    keras_model.add(keras.layers.Activation("linear"))
    keras_model.add(keras.layers.Dense(hidden_size, activation=act_f))
    if eclaire_cfg.num_layers == 2:
        keras_model.add(keras.layers.Dense(hidden_size, activation=act_f))
    keras_model.add(keras.layers.Dense(output_size))
    keras_model.add(keras.layers.Softmax())
    # set weights
    keras_model.set_weights(weights)

    test_input = tf.convert_to_tensor(input_vector.numpy().astype(np.float32))
    out = keras_model.predict(test_input)

    # load data
    train_x = np.load(os.path.join(eclaire_dir, eclaire_cfg.obs_filename))
    if len(train_x.shape) == 3:
        train_x = train_x.squeeze(axis=1) # dims: (N, 1, M) -> (N, M)
    #import ipdb; ipdb.set_trace()
    num_samples = min(eclaire_cfg.num_samples, train_x.shape[0])
    train_x = train_x[:num_samples]
    print("train_x.shape: ", train_x.shape)
    input = tf.convert_to_tensor(train_x.astype(np.float32))
    print("len(fnames): ", len(fnames))
    ruleset = eclaire.extract_rules(keras_model, input, feature_names=fnames, output_class_names=actions)
    ruleset.to_file(os.path.join(eclaire_dir, eclaire_cfg.rule_filename))
