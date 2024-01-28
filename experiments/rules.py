from remix import eclaire, deep_red_c5
import torch
import numpy as np
import tensorflow as tf
from algos import networks
from tensorflow import keras
#from tensorflow.keras.models import Dense, InputLayer
from fnames import get_actions_for_game, get_fnames_for_game
GAME_ENV = "freeway"

data_base_path = "remix_data/" + GAME_ENV + "/"
ruleset_fname = "output.rules"

fnames = get_fnames_for_game(GAME_ENV)
actions = get_actions_for_game(GAME_ENV)

# get model architecture variables
input_size = len(fnames)
output_size = len(actions)
hidden_size = int(2/3 * (output_size + input_size))
act_f = "relu"
# load weights
checkpoint = torch.load(data_base_path + "checkpoint.pth")
state_dict = checkpoint["policy"]
# load data
train_x = np.load(data_base_path + "obs.npy")

# load pytorch model
policy_net = networks.PolicyNet(input_size, hidden_size, output_size, act_f)
policy_net.load_state_dict(state_dict)


# define keras model
keras_model = keras.Sequential()
keras_model.add(keras.Input(shape=(input_size,)))
keras_model.add(keras.layers.Activation("linear"))
keras_model.add(keras.layers.Dense(hidden_size, activation=act_f))
keras_model.add(keras.layers.Dense(output_size))
keras_model.add(keras.layers.Softmax())

# set weights
keras_model.set_weights([state_dict[key].cpu().numpy().T for key in state_dict.keys()])

# print summary
keras_model.summary()

#compare outputs with test input
np_array = train_x[0].astype(np.float32).reshape(1, -1)
test_input_pytorch = torch.tensor(np_array)
test_input_tf = tf.convert_to_tensor(np_array)

out_pytorch = policy_net(test_input_pytorch)
print("pytorch: ", out_pytorch)
out_tf = keras_model.predict(test_input_tf)
print("tf: ", out_tf)


input = tf.convert_to_tensor(train_x.astype(np.float32))
ruleset = eclaire.extract_rules(keras_model, input, feature_names=fnames, output_class_names=actions)
ruleset.to_file(data_base_path + ruleset_fname)
#print(ruleset)



# ###CODE FROM PAST EXPERIMENTS###
#keras_model.set_weights([state_dict[key].cpu().numpy().T for key in state_dict.keys()])
##print(keras_model.layers)
##print(state_dict.keys())
##print(keras_model.layers[0].weights[0].shape)
##print(state_dict["hlayer1.weight"].shape)
##keras_model.layers[0].set_weights(state_dict["hlayer1.weight"].T)
##exit()
##for kl, lw in zip(keras_model.layers, state_dict):
##   kl.set_weights(lw)
#
#
#keras_model.summary()
##exit()
##dummy_input = torch.tensor(np.random.rand(29).astype(np.float32)).unsqueeze(0)
##print(dummy_input)
##out = policy_net(dummy_input)
##print(out)
##torch.onnx.export(policy_net, dummy_input, 'model_simple.onnx', input_names=['input'], output_names=['output'])
#
##model_onnx = onnx.load('model_simple.onnx')
##tf_rep = prepare(model_onnx)
##tf_rep.export_graph('model_simple.pb')
#
##keras_model = onnx_to_keras(model_onnx, ["input"])
#
#
#
#input = tf.convert_to_tensor(train_x.astype(np.float32))
#test_input = tf.convert_to_tensor(np.random.rand(1,29).astype(np.float32))
##y = keras_model.predict(tf.convert_to_tensor(np.random.rand(1,29).astype(np.float32)))
##print(y)
##exit()
##
#
#
#ruleset = eclaire.extract_rules(keras_model, input, feature_names=fnames, output_class_names=actions)
#ruleset.to_file(data_base_path + ruleset_fname)
##ruleset = deep_red_c5.extract_rules(keras_model, input)
#print(ruleset)
#exit()
#ruleset.predict(test_input)
#y_pred, explanations, scores = ruleset.predict_and_explain(test_input)
#
##print(y_pred)
##print(explanations[0][0])
##print(scores)