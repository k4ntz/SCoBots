from remix import eclaire, deep_red_c5
import torch
import numpy as np
import tensorflow as tf
from algos import networks
from tensorflow import keras
#from tensorflow.keras.models import Dense, InputLayer

GAME_ENV = "pong_enemy"

data_base_path = "remix_data/" + GAME_ENV + "/"
ruleset_fname = "output.rules"

fnames_skiing = ['Player1.x', 'Player1.y', 'Mogul1.x', 'Mogul1.y', 'Flag1.x', 'Flag1.y', 'Flag2.x', 'Flag2.y', 'Tree1.x', 'Tree1.y', 'Tree2.x', 'Tree2.y', 'O(Player1)', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Flag1.x', 'Flag1.y', 'Flag1.x[t-1]', 'Flag1.y[t-1]', 'D(Player1,Flag1).x', 'D(Player1,Flag1).y', 'C(Flag1,Flag2).x', 'C(Flag1,Flag2).y', 'DV(Player1).x', 'DV(Player1).y', 'DV(Flag1).x', 'DV(Flag1).y']
fnames_pong = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Ball1,Ball1).x', 'LT(Ball1,Ball1).y', 'LT(Ball1,Enemy1).x', 'LT(Ball1,Enemy1).y', 'LT(Ball1,Player1).x', 'LT(Ball1,Player1).y', 'LT(Enemy1,Ball1).x', 'LT(Enemy1,Ball1).y', 'LT(Enemy1,Enemy1).x', 'LT(Enemy1,Enemy1).y', 'LT(Enemy1,Player1).x', 'LT(Enemy1,Player1).y', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'LT(Player1,Enemy1).x', 'LT(Player1,Enemy1).y', 'LT(Player1,Player1).x', 'LT(Player1,Player1).y', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']
fnames_pong_no_enemy = ['Ball1.x', 'Ball1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'DV(Ball1).x', 'DV(Ball1).y', 'DV(Player1).x', 'DV(Player1).y']
fnames_pong_pruned_no_lt = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']
fnames = fnames_pong_pruned_no_lt

actions_skiing = ["NOOP", "RIGHT", "LEFT"]
actions_pong = ['NOOP', 'FIRE', 'UP', 'DOWN']
actions = actions_pong

# TODO: dont hardcode network architecture
input_size = 33
hidden_size = 24
output_size = 4
act_f = "relu"

policy_net = networks.PolicyNet(input_size, hidden_size, output_size, act_f)

train_x = np.load(data_base_path + "obs.npy")
checkpoint = torch.load(data_base_path + "checkpoint.pth")
state_dict = checkpoint["policy"]
policy_net.load_state_dict(checkpoint["policy"])

keras_model = keras.Sequential()
keras_model.add(keras.Input(shape=(input_size,)))
keras_model.add(keras.layers.Activation("linear"))
keras_model.add(keras.layers.Dense(hidden_size, activation=act_f))
keras_model.add(keras.layers.Dense(output_size))
keras_model.add(keras.layers.Softmax())


keras_model.set_weights([state_dict[key].cpu().numpy().T for key in state_dict.keys()])
#print(keras_model.layers)
#print(state_dict.keys())
#print(keras_model.layers[0].weights[0].shape)
#print(state_dict["hlayer1.weight"].shape)
#keras_model.layers[0].set_weights(state_dict["hlayer1.weight"].T)
#exit()
#for kl, lw in zip(keras_model.layers, state_dict):
#   kl.set_weights(lw)


keras_model.summary()
#exit()
#dummy_input = torch.tensor(np.random.rand(29).astype(np.float32)).unsqueeze(0)
#print(dummy_input)
#out = policy_net(dummy_input)
#print(out)
#torch.onnx.export(policy_net, dummy_input, 'model_simple.onnx', input_names=['input'], output_names=['output'])

#model_onnx = onnx.load('model_simple.onnx')
#tf_rep = prepare(model_onnx)
#tf_rep.export_graph('model_simple.pb')

#keras_model = onnx_to_keras(model_onnx, ["input"])



input = tf.convert_to_tensor(train_x.astype(np.float32))
test_input = tf.convert_to_tensor(np.random.rand(1,29).astype(np.float32))
#y = keras_model.predict(tf.convert_to_tensor(np.random.rand(1,29).astype(np.float32)))
#print(y)
#exit()
#


ruleset = eclaire.extract_rules(keras_model, input, feature_names=fnames, output_class_names=actions)
ruleset.to_file(data_base_path + ruleset_fname)
#ruleset = deep_red_c5.extract_rules(keras_model, input)
print(ruleset)
exit()
ruleset.predict(test_input)
y_pred, explanations, scores = ruleset.predict_and_explain(test_input)

#print(y_pred)
#print(explanations[0][0])
#print(scores)