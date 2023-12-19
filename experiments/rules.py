from remix import eclaire, deep_red_c5
import torch
import numpy as np
import tensorflow as tf
from algos import networks
from tensorflow import keras
#from tensorflow.keras.models import Dense, InputLayer

GAME_ENV = "kangaroo"

data_base_path = "remix_data/" + GAME_ENV + "/"
ruleset_fname = "output.rules"

fnames_skiing = ['Player1.x', 'Player1.y', 'Mogul1.x', 'Mogul1.y', 'Flag1.x', 'Flag1.y', 'Flag2.x', 'Flag2.y', 'Tree1.x', 'Tree1.y', 'Tree2.x', 'Tree2.y', 'O(Player1)', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Flag1.x', 'Flag1.y', 'Flag1.x[t-1]', 'Flag1.y[t-1]', 'D(Player1,Flag1).x', 'D(Player1,Flag1).y', 'C(Flag1,Flag2).x', 'C(Flag1,Flag2).y', 'DV(Player1).x', 'DV(Player1).y', 'DV(Flag1).x', 'DV(Flag1).y']
fnames_pong = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Ball1,Ball1).x', 'LT(Ball1,Ball1).y', 'LT(Ball1,Enemy1).x', 'LT(Ball1,Enemy1).y', 'LT(Ball1,Player1).x', 'LT(Ball1,Player1).y', 'LT(Enemy1,Ball1).x', 'LT(Enemy1,Ball1).y', 'LT(Enemy1,Enemy1).x', 'LT(Enemy1,Enemy1).y', 'LT(Enemy1,Player1).x', 'LT(Enemy1,Player1).y', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'LT(Player1,Enemy1).x', 'LT(Player1,Enemy1).y', 'LT(Player1,Player1).x', 'LT(Player1,Player1).y', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']
fnames_pong_no_enemy = ['Ball1.x', 'Ball1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'LT(Player1,Ball1).x', 'LT(Player1,Ball1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'DV(Ball1).x', 'DV(Ball1).y', 'DV(Player1).x', 'DV(Player1).y']
fnames_pong_pruned_no_lt = ['Ball1.x', 'Ball1.y', 'Enemy1.x', 'Enemy1.y', 'Player1.x', 'Player1.y', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'Enemy1.x', 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'D(Ball1,Enemy1).x', 'D(Ball1,Enemy1).y', 'D(Ball1,Player1).x', 'D(Ball1,Player1).y', 'D(Enemy1,Ball1).x', 'D(Enemy1,Ball1).y', 'D(Enemy1,Player1).x', 'D(Enemy1,Player1).y', 'D(Player1,Ball1).x', 'D(Player1,Ball1).y', 'D(Player1,Enemy1).x', 'D(Player1,Enemy1).y', 'V(Ball1).x', 'V(Enemy1).x', 'V(Player1).x']
fnames_freeway = ['Chicken1.x', 'Chicken1.y', 'Car1.x', 'Car1.y', 'Car2.x', 'Car2.y', 'Car3.x', 'Car3.y', 'Car4.x', 'Car4.y', 'Chicken1.x', 'Chicken1.y', 'Chicken1.x[t-1]', 'Chicken1.y[t-1]', 'Car1.x', 'Car1.y', 'Car1.x[t-1]', 'Car1.y[t-1]', 'Car2.x', 'Car2.y', 'Car2.x[t-1]', 'Car2.y[t-1]', 'Car3.x', 'Car3.y', 'Car3.x[t-1]', 'Car3.y[t-1]', 'Car4.x', 'Car4.y', 'Car4.x[t-1]', 'Car4.y[t-1]', 'LT(Chicken1,Chicken1).x', 'LT(Chicken1,Chicken1).y', 'LT(Chicken1,Car1).x', 'LT(Chicken1,Car1).y', 'LT(Chicken1,Car2).x', 'LT(Chicken1,Car2).y', 'LT(Chicken1,Car3).x', 'LT(Chicken1,Car3).y', 'LT(Chicken1,Car4).x', 'LT(Chicken1,Car4).y', 'LT(Car1,Chicken1).x', 'LT(Car1,Chicken1).y', 'LT(Car1,Car1).x', 'LT(Car1,Car1).y', 'LT(Car1,Car2).x', 'LT(Car1,Car2).y', 'LT(Car1,Car3).x', 'LT(Car1,Car3).y', 'LT(Car1,Car4).x', 'LT(Car1,Car4).y', 'LT(Car2,Chicken1).x', 'LT(Car2,Chicken1).y', 'LT(Car2,Car1).x', 'LT(Car2,Car1).y', 'LT(Car2,Car2).x', 'LT(Car2,Car2).y', 'LT(Car2,Car3).x', 'LT(Car2,Car3).y', 'LT(Car2,Car4).x', 'LT(Car2,Car4).y', 'LT(Car3,Chicken1).x', 'LT(Car3,Chicken1).y', 'LT(Car3,Car1).x', 'LT(Car3,Car1).y', 'LT(Car3,Car2).x', 'LT(Car3,Car2).y', 'LT(Car3,Car3).x', 'LT(Car3,Car3).y', 'LT(Car3,Car4).x', 'LT(Car3,Car4).y', 'LT(Car4,Chicken1).x', 'LT(Car4,Chicken1).y', 'LT(Car4,Car1).x', 'LT(Car4,Car1).y', 'LT(Car4,Car2).x', 'LT(Car4,Car2).y', 'LT(Car4,Car3).x', 'LT(Car4,Car3).y', 'LT(Car4,Car4).x', 'LT(Car4,Car4).y', 'D(Chicken1,Car1).x', 'D(Chicken1,Car1).y', 'D(Chicken1,Car2).x', 'D(Chicken1,Car2).y', 'D(Chicken1,Car3).x', 'D(Chicken1,Car3).y', 'D(Chicken1,Car4).x', 'D(Chicken1,Car4).y', 'D(Car1,Chicken1).x', 'D(Car1,Chicken1).y', 'D(Car1,Car2).x', 'D(Car1,Car2).y', 'D(Car1,Car3).x', 'D(Car1,Car3).y', 'D(Car1,Car4).x', 'D(Car1,Car4).y', 'D(Car2,Chicken1).x', 'D(Car2,Chicken1).y', 'D(Car2,Car1).x', 'D(Car2,Car1).y', 'D(Car2,Car3).x', 'D(Car2,Car3).y', 'D(Car2,Car4).x', 'D(Car2,Car4).y', 'D(Car3,Chicken1).x', 'D(Car3,Chicken1).y', 'D(Car3,Car1).x', 'D(Car3,Car1).y', 'D(Car3,Car2).x', 'D(Car3,Car2).y', 'D(Car3,Car4).x', 'D(Car3,Car4).y', 'D(Car4,Chicken1).x', 'D(Car4,Chicken1).y', 'D(Car4,Car1).x', 'D(Car4,Car1).y', 'D(Car4,Car2).x', 'D(Car4,Car2).y', 'D(Car4,Car3).x', 'D(Car4,Car3).y', 'V(Chicken1).x', 'V(Car1).x', 'V(Car2).x', 'V(Car3).x', 'V(Car4).x']
fnames_bowling = ['Player1.x', 'Player1.y', 'Pin1.x', 'Pin1.y', 'Pin2.x', 'Pin2.y', 'Pin3.x', 'Pin3.y', 'Pin4.x', 'Pin4.y', 'Ball1.x', 'Ball1.y', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Pin1.x', 'Pin1.y', 'Pin1.x[t-1]', 'Pin1.y[t-1]', 'Pin2.x', 'Pin2.y', 'Pin2.x[t-1]', 'Pin2.y[t-1]', 'Pin3.x', 'Pin3.y', 'Pin3.x[t-1]', 'Pin3.y[t-1]', 'Pin4.x', 'Pin4.y', 'Pin4.x[t-1]', 'Pin4.y[t-1]', 'Ball1.x', 'Ball1.y', 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'ED(Player1,Pin1)', 'ED(Player1,Pin2)', 'ED(Player1,Pin3)', 'ED(Player1,Pin4)', 'ED(Player1,Ball1)', 'ED(Pin1,Player1)', 'ED(Pin1,Pin2)', 'ED(Pin1,Pin3)', 'ED(Pin1,Pin4)', 'ED(Pin1,Ball1)', 'ED(Pin2,Player1)', 'ED(Pin2,Pin1)', 'ED(Pin2,Pin3)', 'ED(Pin2,Pin4)', 'ED(Pin2,Ball1)', 'ED(Pin3,Player1)', 'ED(Pin3,Pin1)', 'ED(Pin3,Pin2)', 'ED(Pin3,Pin4)', 'ED(Pin3,Ball1)', 'ED(Pin4,Player1)', 'ED(Pin4,Pin1)', 'ED(Pin4,Pin2)', 'ED(Pin4,Pin3)', 'ED(Pin4,Ball1)', 'ED(Ball1,Player1)', 'ED(Ball1,Pin1)', 'ED(Ball1,Pin2)', 'ED(Ball1,Pin3)', 'ED(Ball1,Pin4)']
fnames_kangaroo_pruned = ['Player1.x', 'Player1.y', 'Monkey1.x', 'Monkey1.y', 'Monkey2.x', 'Monkey2.y', 'Fruit1.x', 'Fruit1.y', 'Fruit2.x', 'Fruit2.y', 'Bell1.x', 'Bell1.y', 'Child1.x', 'Child1.y', 'Ladder1.x', 'Ladder1.y', 'Ladder2.x', 'Ladder2.y', 'Platform1.x', 'Platform1.y', 'Platform2.x', 'Platform2.y', 'FallingCoconut1.x', 'FallingCoconut1.y', 'ThrownCoconut1.x', 'ThrownCoconut1.y', 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Monkey1.x', 'Monkey1.y', 'Monkey1.x[t-1]', 'Monkey1.y[t-1]', 'Monkey2.x', 'Monkey2.y', 'Monkey2.x[t-1]', 'Monkey2.y[t-1]', 'Fruit1.x', 'Fruit1.y', 'Fruit1.x[t-1]', 'Fruit1.y[t-1]', 'Fruit2.x', 'Fruit2.y', 'Fruit2.x[t-1]', 'Fruit2.y[t-1]', 'Bell1.x', 'Bell1.y', 'Bell1.x[t-1]', 'Bell1.y[t-1]', 'Child1.x', 'Child1.y', 'Child1.x[t-1]', 'Child1.y[t-1]', 'Ladder1.x', 'Ladder1.y', 'Ladder1.x[t-1]', 'Ladder1.y[t-1]', 'Ladder2.x', 'Ladder2.y', 'Ladder2.x[t-1]', 'Ladder2.y[t-1]', 'Platform1.x', 'Platform1.y', 'Platform1.x[t-1]', 'Platform1.y[t-1]', 'Platform2.x', 'Platform2.y', 'Platform2.x[t-1]', 'Platform2.y[t-1]', 'FallingCoconut1.x', 'FallingCoconut1.y', 'FallingCoconut1.x[t-1]', 'FallingCoconut1.y[t-1]', 'ThrownCoconut1.x', 'ThrownCoconut1.y', 'ThrownCoconut1.x[t-1]', 'ThrownCoconut1.y[t-1]', 'D(Player1,Monkey1).x', 'D(Player1,Monkey1).y', 'D(Player1,Monkey2).x', 'D(Player1,Monkey2).y', 'D(Player1,Fruit1).x', 'D(Player1,Fruit1).y', 'D(Player1,Fruit2).x', 'D(Player1,Fruit2).y', 'D(Player1,Bell1).x', 'D(Player1,Bell1).y', 'D(Player1,Child1).x', 'D(Player1,Child1).y', 'D(Player1,Ladder1).x', 'D(Player1,Ladder1).y', 'D(Player1,Ladder2).x', 'D(Player1,Ladder2).y', 'D(Player1,Platform1).x', 'D(Player1,Platform1).y', 'D(Player1,Platform2).x', 'D(Player1,Platform2).y', 'D(Player1,FallingCoconut1).x', 'D(Player1,FallingCoconut1).y', 'D(Player1,ThrownCoconut1).x', 'D(Player1,ThrownCoconut1).y', 'D(Monkey1,Monkey2).x', 'D(Monkey1,Monkey2).y', 'D(Monkey1,Fruit1).x', 'D(Monkey1,Fruit1).y', 'D(Monkey1,Fruit2).x', 'D(Monkey1,Fruit2).y', 'D(Monkey1,Bell1).x', 'D(Monkey1,Bell1).y', 'D(Monkey1,Child1).x', 'D(Monkey1,Child1).y', 'D(Monkey1,Ladder1).x', 'D(Monkey1,Ladder1).y', 'D(Monkey1,Ladder2).x', 'D(Monkey1,Ladder2).y', 'D(Monkey1,Platform1).x', 'D(Monkey1,Platform1).y', 'D(Monkey1,Platform2).x', 'D(Monkey1,Platform2).y', 'D(Monkey1,FallingCoconut1).x', 'D(Monkey1,FallingCoconut1).y', 'D(Monkey1,ThrownCoconut1).x', 'D(Monkey1,ThrownCoconut1).y', 'D(Monkey2,Fruit1).x', 'D(Monkey2,Fruit1).y', 'D(Monkey2,Fruit2).x', 'D(Monkey2,Fruit2).y', 'D(Monkey2,Bell1).x', 'D(Monkey2,Bell1).y', 'D(Monkey2,Child1).x', 'D(Monkey2,Child1).y', 'D(Monkey2,Ladder1).x', 'D(Monkey2,Ladder1).y', 'D(Monkey2,Ladder2).x', 'D(Monkey2,Ladder2).y', 'D(Monkey2,Platform1).x', 'D(Monkey2,Platform1).y', 'D(Monkey2,Platform2).x', 'D(Monkey2,Platform2).y', 'D(Monkey2,FallingCoconut1).x', 'D(Monkey2,FallingCoconut1).y', 'D(Monkey2,ThrownCoconut1).x', 'D(Monkey2,ThrownCoconut1).y', 'DV(Player1).x', 'DV(Player1).y', 'DV(Monkey1).x', 'DV(Monkey1).y', 'DV(Monkey2).x', 'DV(Monkey2).y', 'DV(Fruit1).x', 'DV(Fruit1).y', 'DV(Fruit2).x', 'DV(Fruit2).y', 'DV(Bell1).x', 'DV(Bell1).y', 'DV(Child1).x', 'DV(Child1).y', 'DV(FallingCoconut1).x', 'DV(FallingCoconut1).y', 'DV(ThrownCoconut1).x', 'DV(ThrownCoconut1).y']

fnames = fnames_kangaroo_pruned

actions_skiing = ["NOOP", "RIGHT", "LEFT"]
actions_pong = ['NOOP', 'FIRE', 'UP', 'DOWN']
actions_freeway = ['NOOP', 'UP', 'DOWN']
actions_bowling = ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE']
actions_kangaroo_pruned = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT']
actions = actions_kangaroo_pruned


input_size = len(fnames)
output_size = len(actions)
hidden_size = int(2/3 * (output_size + input_size))
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